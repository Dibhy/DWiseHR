from flask import Flask, request, render_template
import os
from werkzeug.utils import secure_filename
import docx2txt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import __version__ as sklearn_version
import joblib
from warnings import filterwarnings

filterwarnings('ignore')

app = Flask(__name__)

class ResumeMatcherApp:
    def __init__(self):
        self.resumes = []
        self.job_text = None
        self.cv_filename = None
        self.result_texts = []

    def load_resume(self, file_path):
        try:
            if file_path.endswith(('.docx', '.pdf')):
                if file_path.endswith('.docx'):
                    resume_text = docx2txt.process(file_path)
                else:
                    resume_text = self.read_text_from_pdf(file_path)

                self.resumes.append({'path': file_path, 'text': resume_text})
        except Exception as e:
            print(f"Error loading resume '{file_path}': {e}")

    def load_job(self, file_path):
        try:
            if file_path.endswith('.docx'):
                self.job_text = docx2txt.process(file_path)
            else:
                print("Unsupported file format for job description.")
        except Exception as e:
            print(f"Error loading job description: {e}")

    def calculate_similarity(self):
        try:
            if self.job_text is not None and self.resumes:
                text = [resume['text'] for resume in self.resumes] + [self.job_text]
                cv = CountVectorizer()
                count_matrix = cv.fit_transform(text)
                similarity_scores = cosine_similarity(count_matrix[:-1], count_matrix[-1])

                # Sort resumes by similarity score in descending order
                sorted_resumes = sorted(zip(self.resumes, similarity_scores), key=lambda x: x[1], reverse=True)

                self.result_texts = []

                for resume, similarity_score in sorted_resumes:
                    resume_path = os.path.basename(resume['path'])  # Get only the filename
                    match_percentage = similarity_score.item() * 100  # Convert to Python float
                    result_text = f'Resume: {resume_path}\n'
                    result_text += f'Similarity score: {match_percentage:.2f}%\n'
                    result_text += f'Your Resume is {match_percentage:.2f}% match to the job description!\n\n'
                    self.result_texts.append(result_text)

                    # Print similarity score using .item()
                    print('\nSimilarity score: {}%'.format(match_percentage))

                # Save the model using joblib
                if sklearn_version >= '0.24.0':
                    self.cv_filename = 'model.pkl'
                    joblib.dump(cv, self.cv_filename)
                else:
                    print('Please upgrade scikit-learn to version 0.24.0 or newer to use joblib directly.')

        except Exception as e:
            print(f"Error calculating similarity: {e}")

    def read_text_from_pdf(self, file_path):
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfFileReader(file)
                text = ""
                for page_num in range(pdf_reader.numPages):
                    text += pdf_reader.getPage(page_num).extractText()
                return text
        except Exception as e:
            print(f"Error reading PDF file '{file_path}': {e}")
            return ""

# Create an instance of the app
resume_app = ResumeMatcherApp()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            if 'job' in request.files:
                job_file = request.files['job']
                if job_file.filename == '':
                    return render_template('index.html', result_texts=["Please select a job description file."])

                job_path = os.path.join("uploads", secure_filename(job_file.filename))
                job_file.save(job_path)
                resume_app.load_job(job_path)

                if 'resume' in request.files:
                    resume_files = request.files.getlist('resume')
                    for resume_file in resume_files:
                        if resume_file.filename != '':
                            resume_path = os.path.join("uploads", secure_filename(resume_file.filename))
                            resume_file.save(resume_path)
                            resume_app.load_resume(resume_path)

                resume_app.calculate_similarity()

                return render_template('result.html', result_texts=resume_app.result_texts)

        except Exception as e:
            print(f"Error processing files: {e}")
            return render_template('index.html', result_texts=["An error occurred while processing files."])

    return render_template('index.html', result_texts=None)

if __name__ == '__main__':
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)
