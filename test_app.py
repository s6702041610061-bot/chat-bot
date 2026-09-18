"""Streamlit integration checks: Gemini and dotenv are mocked; no API calls."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from streamlit.testing.v1 import AppTest
import rag


class AppTests(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()
        self.env = patch.dict('os.environ', {'GEMINI_API_KEY_INSURVERSE': 'offline-test-placeholder'})
        self.env.start()
        self.dotenv = patch('dotenv.load_dotenv')
        self.dotenv.start()
        self.client_factory = patch('gemini_service.create_client')
        self.client_factory_mock = self.client_factory.start()
        self.client = self.client_factory_mock.return_value
        self.embed_mock = self.client.models.embed_content
        self.embed_mock.side_effect = AssertionError('Unexpected API')
        self.chat = self.client.chats.create.return_value
        self.chat.send_message_stream.return_value = iter([SimpleNamespace(text='คำตอบทดสอบ')])
        self.addCleanup(patch.stopall)

    def app(self):
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
        self.assertFalse(app.exception)
        return app

    def test_startup_no_api_and_direct(self):
        app = self.app()
        self.embed_mock.assert_not_called()
        app.chat_input[0].set_value('หลักสูตรนี้มีชื่อภาษาไทยว่าอะไร?').run()
        self.assertFalse(app.exception)
        self.chat.send_message_stream.assert_not_called()
        self.embed_mock.assert_not_called()
        self.assertIn('FAQ 001', app.session_state['messages'][-1]['content'])

    def test_whitespace_question_is_not_processed(self):
        app = self.app()
        original_count = len(app.session_state['messages'])
        app.chat_input[0].set_value('   ').run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state['messages']), original_count)
        self.chat.send_message_stream.assert_not_called()
        self.embed_mock.assert_not_called()

    def test_teacher_count_uses_faq_without_gemini(self):
        app = self.app()
        app.chat_input[0].set_value('ภาคนี้มีอาจารย์กี่คน').run()
        self.assertFalse(app.exception)
        answer = app.session_state['messages'][-1]['content']
        self.assertIn('20 คน', answer)
        self.assertIn('FAQ 111', answer)
        self.chat.send_message_stream.assert_not_called()
        self.embed_mock.assert_not_called()

    def test_local_stream_and_history(self):
        app = self.app()
        app.chat_input[0].set_value('ค่าเล่าเรียนเท่าไหร่').run()
        self.assertFalse(app.exception)
        self.chat.send_message_stream.assert_called_once()
        self.embed_mock.assert_not_called()
        prompt = self.chat.send_message_stream.call_args.args[0]
        self.assertEqual(prompt.count('ค่าเล่าเรียนเท่าไหร่'), 1)
        self.assertEqual(prompt.count('## FAQ'), 3)
        self.chat.send_message_stream.assert_called_once()
        self.assertEqual(app.session_state['messages'][-1]['content'], 'คำตอบทดสอบ')
        self.assertEqual(self.client.chats.create.call_args.kwargs['history'], [])

    def test_short_follow_up_uses_remembered_faq_context(self):
        app = self.app()
        app.chat_input[0].set_value('หลักสูตรนี้มีกี่แขนงวิชาให้เลือก?').run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state[rag.CONTEXT_STATE_KEY]['topic'], 'แขนงวิชา')

        self.chat.send_message_stream.reset_mock()
        app.chat_input[0].set_value('อะไรบ้าง').run()
        self.assertFalse(app.exception)
        prompt = self.chat.send_message_stream.call_args.args[0]
        self.assertIn('## FAQ 020', prompt)
        self.assertIn('## FAQ 021', prompt)
        self.assertIn('## FAQ 022', prompt)
        self.assertEqual(app.session_state[rag.CONTEXT_STATE_KEY]['topic'], 'แขนงวิชา')

    def test_clear_history_also_clears_retrieval_context(self):
        app = self.app()
        app.chat_input[0].set_value('หลักสูตรนี้มีกี่แขนงวิชาให้เลือก?').run()
        self.assertIn(rag.CONTEXT_STATE_KEY, app.session_state)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertNotIn(rag.CONTEXT_STATE_KEY, app.session_state)

    def test_generation_failure_clears_partial(self):
        def broken():
            yield SimpleNamespace(text='partial')
            raise RuntimeError('simulated')
        self.chat.send_message_stream.return_value = broken()
        app = self.app()
        app.chat_input[0].set_value('ค่าเล่าเรียนเท่าไหร่').run()
        self.assertFalse(app.exception)
        self.assertNotIn('partial', app.session_state['messages'][-1]['content'])
        self.assertNotIn('▌', app.session_state['messages'][-1]['content'])

    def test_no_key_still_direct(self):
        with patch.dict('os.environ', {'GEMINI_API_KEY_INSURVERSE': ''}), patch('streamlit.secrets', {}):
            app = self.app()
            app.chat_input[0].set_value('หลักสูตรนี้มีชื่อภาษาไทยว่าอะไร?').run()
            self.assertFalse(app.exception)
            self.assertIn('FAQ 001', app.session_state['messages'][-1]['content'])
            self.chat.send_message_stream.assert_not_called()

    def test_exact_faq_direct_accuracy(self):
        import re
        retriever = rag.build_retriever((ROOT / 'FAQ_Chatbot_100.md').read_bytes(), ROOT)
        direct_count = 0
        for index, chunk in enumerate(retriever['chunks']):
            question = re.search(r'\*\*คำถาม:\*\*\s*(.+)', chunk).group(1)
            scores, indexes = rag.search_tfidf(question, retriever)
            if rag.choose_route(scores, question) == 'A':
                direct_count += 1
                self.assertEqual(int(indexes[0]), index)
        self.assertGreater(direct_count, 0)


if __name__ == '__main__':
    unittest.main()
