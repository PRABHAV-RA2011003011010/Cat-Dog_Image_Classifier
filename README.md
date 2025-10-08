# Cat-Dog_Image_Classifier
## Steps To execute the Project:
->After Pulling the repo, Create a virtual environment:<br>
python -m venv env<br>
->Activate the env:<br>
env\Scripts\activate<br>
->Install the required dependencies:<br>
pip install -r requirements.txt or pip install streamlit tensorflow pillow numpy huggingface_hub<br>
->First run fast api:<br>
uvicorn api:app --reload(if not working manually type the command)<br>
->To run the stream lit application:<br>
streamlit run app.py<br>
