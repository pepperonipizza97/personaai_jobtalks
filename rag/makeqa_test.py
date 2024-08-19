import os
import fitz  # PyMuPDF의 모듈명
import nltk
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

# PDF에서 텍스트를 추출하는 함수
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# 텍스트를 문장으로 나누는 함수
def split_text_into_sentences(text):
    nltk.download('punkt')
    sentences = nltk.sent_tokenize(text)
    return sentences

# 질문을 생성하는 함수
def generate_qa_pairs(sentences, question_generator):
    qa_pairs = []
    for sentence in sentences:
        question = question_generator(sentence)[0]['generated_text']
        qa_pairs.append({'question': question, 'answer': sentence})
    return qa_pairs

# PDF 파일에서 QA 쌍을 생성하는 메인 함수
def generate_qa_from_pdfs(pdf_folder_path):
    # 모델 및 토크나이저 로드
    model_name = "rtzr/ko-gemma-2-9b-it"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Question generation pipeline 초기화
    question_generator = pipeline('text2text-generation', model=model, tokenizer=tokenizer)

    qa_dataset = []
    pdf_files = [f for f in os.listdir(pdf_folder_path) if f.endswith('.pdf')]

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder_path, pdf_file)
        text = extract_text_from_pdf(pdf_path)
        sentences = split_text_into_sentences(text)
        qa_pairs = generate_qa_pairs(sentences, question_generator)
        qa_dataset.extend(qa_pairs)

    return qa_dataset

# 결과 실행
pdf_folder_path = './test_source'  # PDF 파일이 저장된 폴더 경로
qa_dataset = generate_qa_from_pdfs(pdf_folder_path)

# 생성된 QA 쌍 확인
for qa in qa_dataset:
    print(f"Question: {qa['question']}\nAnswer: {qa['answer']}\n")
