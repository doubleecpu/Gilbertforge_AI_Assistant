
import os
import streamlit as st
from pathlib import Path
from Model import Model
from View import App_View
from Coder import Classifier, MetricsComputer, Evaluator, Agent

class GilbertForge_App:
    def __init__(self):
        self.execution_Code = 0
        self.streamlit = st
        self.Appplication_Status = 1
        self.PDF_FOLDER_NAME = "PDF_Folder"
        self.Assets_FOLDER_NAME = "assets"
        self.Path = Path
        self.upload_dir = self.Path(__file__).parent / self.Assets_FOLDER_NAME / self.PDF_FOLDER_NAME
        self.metrics_computer = MetricsComputer(self)
        self.model = Model(self)
        self.classifier = Classifier(self)
        self.evaluator = Evaluator(self)
        self.agent = Agent(self)
        self.app_view = App_View(self)
        self.Appplication_Status = 1
        print(f"GilbertForge Application Initialized [status ={self.Appplication_Status}]")

    def Application_Loop(self):
        self.execution_Code = 1
        while self.execution_Code == 1:
            try:
                self.execution_Code = 1
            except Exception as e:
                print(f"An error occurred: {e}")
                self.execution_Code = 2
        return
    
    def Application_Status(self, status):
        print(f"GilbertForge Application {status} [status ={self.Appplication_Status}]")

app = GilbertForge_App()
if app.Appplication_Status != 1:
    app.Application_Status("running")
else:
    app.app_view.display_app("GilbertForge AI")
    app.Application_Status("is starting")

while True:
    app.Application_Loop()
    if app.Appplication_Status != 1:
        break
# Set status to 0 after normal loop exit
if app.Appplication_Status == 0:
    app.Application_Status("is shutting down")
elif app.Appplication_Status == 2:
    app.Application_Status("Stopped! Error occurred")
