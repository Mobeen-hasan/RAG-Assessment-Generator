import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

class QuestionGenerator:
    def __init__(self, model_provider="gemini", temperature=0.7):
        self.model_provider = model_provider
        self.temperature = temperature
        self.llm = self._initialize_llm()
        
        # 1. Question Generation Prompt
        self.prompt = PromptTemplate(
            input_variables=["difficulty", "question_type", "context"],
            template="""
            You are an expert exam setter. 
            TASK: Generate a single {difficulty} {question_type} question based ONLY on the text below.
            
            CONTEXT:
            {context}
            
            STRICT RULES:
            1. If type is 'Multiple Choice', provide options A, B, C, D.
            2. If type is 'Fill in the Blank', provide a sentence with a missing word represented by '_______'.
            3. If type is 'Long Question', ask a conceptual question requiring a detailed answer.
            4. If type is 'Short Answer', just ask the question.
            5. Use the separator '|||' to hide the answer.
            6. CRITICAL: Put the Explanation on a new line.
            
            OUTPUT FORMAT:
            Question: [The Question Text]
            [Options if MCQ]
            
            |||
            
            Correct Answer: [The Answer]
            
            Explanation: [Why this is correct]
            """
        )

        # 2. Grading Prompt
        self.grading_prompt = PromptTemplate(
            input_variables=["question", "correct_answer", "user_answer"],
            template="""
            You are a strict teaching assistant.
            Question: {question}
            Correct Answer: {correct_answer}
            Student Answer: {user_answer}
            
            Grade the student's answer.
            Output Format:
            Grade: [Correct/Partially Correct/Incorrect]
            Feedback: [1 sentence feedback]
            """
        )

    def _initialize_llm(self):
        # 1. Google Gemini 2.0
        if self.model_provider == "gemini":
            return ChatGoogleGenerativeAI(
                model="gemini-2.0-flash", 
                temperature=self.temperature,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
        
        # 2. Llama 3.3 70B (High Quality)
        elif self.model_provider == "groq":
            return ChatGroq(
                model_name="llama-3.3-70b-versatile", 
                temperature=self.temperature,
                groq_api_key=os.getenv("GROQ_API_KEY")
            )
            
        # 3. Llama 3.1 8B (Super Fast)
        elif self.model_provider == "fast":
            return ChatGroq(
                model_name="llama-3.1-8b-instant", 
                temperature=self.temperature,
                groq_api_key=os.getenv("GROQ_API_KEY")
            )
        
        # Fallback
        else:
            return ChatGroq(
                model_name="llama-3.3-70b-versatile",
                temperature=self.temperature,
                groq_api_key=os.getenv("GROQ_API_KEY")
            )

    def generate(self, context, question_type, difficulty):
        chain = self.prompt | self.llm | StrOutputParser()
        return chain.invoke({
            "context": context,
            "question_type": question_type,
            "difficulty": difficulty
        })

    def check_answer(self, question, correct_answer, user_answer):
        chain = self.grading_prompt | self.llm | StrOutputParser()
        return chain.invoke({
            "question": question,
            "correct_answer": correct_answer,
            "user_answer": user_answer
        })