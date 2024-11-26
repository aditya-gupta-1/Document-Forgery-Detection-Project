import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow INFO and WARNING logs
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # Suppress TensorFlow Warnings

from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from validate_document import validate_document  # Import validate_document function

app = Flask(__name__)

# Set up the upload folder and allowed file extensions
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

# Function to check if the file has a valid extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Home route
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        document_file = request.files.get("document")
        template_file = request.files.get("template")

        if document_file and allowed_file(document_file.filename) and template_file and allowed_file(template_file.filename):
            document_filename = secure_filename(document_file.filename)
            template_filename = secure_filename(template_file.filename)

            # Save the uploaded files
            document_path = os.path.join(app.config['UPLOAD_FOLDER'], document_filename)
            template_path = os.path.join(app.config['UPLOAD_FOLDER'], template_filename)
            
            document_file.save(document_path)
            template_file.save(template_path)

            # Call the validation function
            reference_text = "expectedtextinavaliddocument"  # Replace with actual reference text
            results = validate_document(document_path, template_path, reference_text)

            if results:
                # Return the results to the webpage
                return render_template("index.html", layout_confidence=results['layout_confidence'],
                                       text_similarity=results['text_similarity'],
                                       overall_accuracy=results['overall_accuracy'],
                                       verification_result=results['verification_result'])
            else:
                return render_template("index.html", error="An error occurred during document verification.")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)


       




