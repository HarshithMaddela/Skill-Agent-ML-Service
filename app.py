from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import re

app = Flask(__name__)

print("Loading all-MiniLM-L6-v2 model into memory...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully.")

MASTER_SKILLS = [
    "Java", "Spring Boot", "React", "Python", "Machine Learning", "Deep Learning",
    "SQL", "HTML", "CSS", "JavaScript",
    "Power Systems", "Electrical Machines", "Control Systems", "Circuit Analysis",
    "MATLAB", "Simulink", "Transformers", "Transmission", "Distribution", "Protection Systems",
    "Digital Electronics", "Analog Electronics", "VLSI", "Signal Processing",
    "Communication Systems", "Embedded Systems", "Microcontrollers", "IoT", "FPGA",
    "Structural Engineering", "Geotechnical Engineering", "Fluid Mechanics", "Surveying",
    "Construction Management", "Concrete Technology", "Transportation Engineering"
]

def extract_target_skills(jd_text):
    jd_lower = jd_text.lower()
    return [skill for skill in MASTER_SKILLS if skill.lower() in jd_lower]

def chunk_text(text):
    sentences = re.split(r'(?<=[.!?]) +|\n+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 5]

@app.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.json
    resume_text = data.get('resume_text', '')
    jd_text = data.get('job_description', '')

    if not resume_text or not jd_text:
        return jsonify({'error': 'Missing resume or job description data'}), 400

    target_skills = extract_target_skills(jd_text)

    if not target_skills:
        target_skills = ["Java", "Spring Boot", "React", "SQL"]

    resume_sentences = chunk_text(resume_text)

    if not resume_sentences:
        resume_sentences = [resume_text]

    resume_embeddings = model.encode(resume_sentences, convert_to_tensor=True)

    results = []

    for skill in target_skills:
        skill_context = f"Experience and proficiency with {skill}."
        skill_embedding = model.encode(skill_context, convert_to_tensor=True)

        cosine_scores = util.cos_sim(skill_embedding, resume_embeddings)[0]
        max_score = cosine_scores.max().item()

        scaled_score = min(10, max(1, int((max_score * 10) * 1.5)))

        if skill.lower() in resume_text.lower():
            scaled_score = max(scaled_score, 7)
            scaled_score = min(10, scaled_score + 2)

        results.append({
            "name": skill,
            "score": int(scaled_score)
        })

    return jsonify({"evaluated_skills": results})

if __name__ == '__main__':
    app.run(port=5000, debug=True)