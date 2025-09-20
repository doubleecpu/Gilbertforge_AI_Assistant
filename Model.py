from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.chains import SequentialChain
from pathlib import Path
import os, pickle

"""
Loads all PDF files from the specified folder, splits their contents into text chunks,
saves the processed chunks to a pickle file, and returns the path to the saved file.

Parameters:
    pdf_folder_name (str): The name of the folder containing PDF files to be loaded.

Returns:
    file_out: the pickle file containing the processed text chunks.
"""
class Model:
    def __init__(self, app):
        self.app = app
        self.PDF_FOLDER_NAME = app.PDF_FOLDER_NAME
        self.advanced_summary = None
        self.data_summary = None
        self.question = None
        self.prompt = None
        self.scenario_template = None
        self.LLM_CHAIN= None
        self.PromptTemplate = PromptTemplate
        self.LLMChain = LLMChain
        self.SequentialChain = SequentialChain
        self.ChatOpenAI = ChatOpenAI
        self.PKL_FILE = None
        self.analysis_chain = None
        self.recommend_chain = None
        self.overall_chain = None
        self.chat_model = None
        self.output_key = None
        self.texts = None

    def save_file(self, data, filename):
        with open(filename, "w") as f:
            f.write(data)
        return f"File saved as {filename}"


    def load_pdfs(self, pdf_folder_name):
        print("Reading PDFs from folder:", pdf_folder_name)
        pdf_folder = Path(pdf_folder_name)
        documents = []
        for file in pdf_folder.glob("*.pdf"):
            loader = PyPDFLoader(str(file))
            documents.extend(loader.load())
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts    = splitter.split_documents(documents)
            print("Total chunks:", len(texts))
            print(f"Loaded {len(documents)} pages")
        with open("processed_texts.pkl","wb") as file_out:
            pickle.dump(texts, file_out)
            os.fsync(file_out)
            self.PKL_FILE = file_out
        print("Chunks cached ➜ processed_texts.pkl")
        return file_out

    def analyze_pdfs(self, query):
        """Analyzes the processed PDF texts using a language model to answer the given query.

        """
        self.question = query
        if(self.app.app_view.model_name in ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"]):
            return self.analyze_with_gpt40_mini()
        else:
            print(f"Model {self.app.app_view.model_name} not supported.")

    def analyze_with_gpt40_mini(self):
        with open(self.app.app_view.Processed_Texts_Path, "rb") as self.PKL_FILE:
            self.texts = pickle.load(self.PKL_FILE)
            self.chat_model = self.ChatOpenAI(
            temperature=0.5,
            model_name="gpt-4o-mini",
            openai_api_key=self.app.app_view.key_input
        )
        self.scenario_template = '''
        You are an expert AI sales analyst. Using the data summary below, answer the question with detailed analysis and actionable recommendations.
        {self.advanced_summary}
        Question: {self.question}
        Detailed Analysis and Recommendations:
        '''        
        self.prompt = self.PromptTemplate(
            template=self.scenario_template,
            input_variables=['advanced_summary','question']
        )
        self.LLM_CHAIN = self.LLMChain(llm=self.chat_model, prompt=self.prompt)
        print(self.generate_insight(self.advanced_summary, "Which product should we push next quarter and why?"))
        # Define chains for analysis and recommendations
        # -- analysis chain
        self.analysis_prompt = self.PromptTemplate(
            template="""Analyze the following advanced sales data summary:

        {self.advanced_summary}

        Give a concise list of the three most important insights.""",
            input_variables=['advanced_summary']
        )
        self.analysis_chain = self.LLMChain(llm=self.chat_model, prompt=self.analysis_prompt, output_key='analysis')
        # -- recommendation chain
        self.recommend_prompt = self.PromptTemplate(
            template="""Based on the analysis below, suggest concrete tactics to solve the question: {self.question}

        Analysis:
        {self.analysis}

        Recommendations:""",
            input_variables=['analysis','question']
        )
        self.recommend_chain = self.LLMChain(llm=self.chat_model, prompt=self.recommend_prompt, output_key='recommendations')
        overall_chain = self.SequentialChain(
            chains=[self.analysis_chain, self.recommend_chain],
            input_variables=['advanced_summary','question'],
            output_variables=['analysis','recommendations'],
            verbose=True
        )
        self.overall_chain = overall_chain({'advanced_summary': self.advanced_summary, 'question': self.question})
        return self.overall_chain['analysis'] + '\n\n' + self.overall_chain['recommendations']

    def generate_insight(self, summary, question):

        return self.LLM_CHAIN.run(advanced_summary=summary, question=question)

    def ask_bi_assistant(self, question:str):
        self.question = question
        template = """
        You are an expert business intelligence assistant. Using the data summary below, answer the question with detailed analysis and actionable recommendations.
        {self.data_summary}
        Question: {self.question}
        Detailed Analysis and Recommendations:
        """
        self.prompt = self.PromptTemplate(
            input_variables=["data_summary", "question"],
            template=template,
        )
        self.LLM_CHAIN = self.LLMChain(llm=ChatOpenAI(temperature=0), prompt=self.prompt)
        # Example data summary (replace with actual data summary)
        self.data_summary = "The sales increased by 20% in the last quarter, with a significant rise in product A."
        return self.LLM_CHAIN.run(data_summary=self.data_summary, question=self.question)
