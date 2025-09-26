import chromadb
from langchain_huggingface import HuggingFaceEndpointEmbeddings

# vector db name
collection_name = "aou_faq_collection"
client_file_name = "./chroma"
# info to be embedded
info = """
Question 1: Where is the Arab Open University Oman branch located?
Answer 1: It is located in Al Maabilah, Seeb, Muscat Governorate, Sultanate of Oman (Building No. 994, Way No. 92).

Question 2: When was the Arab Open University Oman branch established?
Answer 2: The Oman branch was established in 2002 as part of the wider Arab Open University network.

Question 3: What type of institution is the Arab Open University Oman?
Answer 3: It is a non-profit, private higher education institution offering open and distance learning.

Question 4: Which authority accredits the Arab Open University Oman?
Answer 4: It is accredited by the Oman Authority for Academic Accreditation and Quality Assurance of Education.

Question 5: What international partnerships does the Arab Open University Oman have?
Answer 5: It has an academic partnership with The Open University in the United Kingdom, which validates many of its programs.

Question 6: What undergraduate programs are offered at the Oman branch?
Answer 6: Undergraduate programs include Business Studies (with specializations such as Management, Marketing, Accounting, Economics, HR, and MIS), Information Technology and Computing (Computer Science, Network & Security, Computing with Business), Law, Early Childhood Education, and English Language Studies.

Question 7: What postgraduate programs are available at the Oman branch?
Answer 7: Postgraduate programs include Master of Education in Instructional Technology, Master of Education in Educational Leadership, and Master of Business Administration (MBA).

Question 8: What is the language of instruction at the Arab Open University Oman?
Answer 8: Programs are taught in both English and Arabic, depending on the faculty and specialization.

Question 9: What is the admission requirement for undergraduate programs?
Answer 9: Applicants must hold a General Secondary School Certificate or its equivalent, with additional requirements depending on the program.

Question 10: What is the admission requirement for postgraduate programs?
Answer 10: Applicants must hold a recognized bachelor’s degree and meet program-specific requirements, such as English proficiency for MBA programs.

Question 11: What is the mode of study at the Arab Open University Oman?
Answer 11: The university follows a blended learning model, combining online learning with face-to-face tutorials.

Question 12: What facilities are available on campus?
Answer 12: Facilities include a library, e-learning systems (LMS), student information system (SIS), training center, research center, and student housing for female students.

Question 13: Does the Arab Open University Oman provide student support services?
Answer 13: Yes, it provides academic advising, career counseling, IT support, and extracurricular activities through student clubs and organizations.

Question 14: What student life opportunities are available?
Answer 14: Students can join cultural clubs, academic groups, sports teams, and community service initiatives. Female students also have access to hostel facilities with transport services.

Question 15: What is the tuition fee structure at the Oman branch?
Answer 15: Tuition fees vary by program, but the admission fee for bachelor’s programs is 46.300 OMR (non-refundable). Fees are generally lower than traditional universities, reflecting AOU’s mission of affordable education.

Question 16: What is the teaching methodology at the Arab Open University Oman?
Answer 16: The methodology emphasizes self-learning, supported by online resources, face-to-face tutorials, and continuous assessment.

Question 17: How long does it take to complete an undergraduate degree?
Answer 17: Most undergraduate programs take four years of full-time study, though part-time and flexible options are available.

Question 18: What is the vision of the Arab Open University Oman?
Answer 18: Its vision is to provide accessible, high-quality education that contributes to Oman’s human and economic development, aligned with Oman Vision 2040.

Question 19: What is the mission of the Arab Open University Oman?
Answer 19: Its mission is to deliver affordable, flexible, and internationally recognized education, while fostering research, innovation, and community service.

Question 20: How can prospective students apply to the Arab Open University Oman?
Answer 20: Applications can be submitted online through the university’s official website (www.aou.edu.om), with required documents uploaded in PDF format and admission fees paid online or in person.
"""


def create_vector_db():
    # embedding model and chroma client (vector db)
    embedding = HuggingFaceEndpointEmbeddings()
    client = chromadb.PersistentClient(client_file_name)
    collection = client.get_or_create_collection(collection_name)

    # converting into Q/A pairs
    lines = info.strip().replace("\n\n", "\n").split("\n")
    questions = lines[0::2]  # even indices: Q
    answers = lines[1::2]  # odd indices: A

    # each Q with its A for embedding, better retrieval
    qa_docs = [f"{q} {a}" for q, a in zip(questions, answers)]

    # sequential string ids: "1", "2", ... for each pair of q/a
    ids = [str(i + 1) for i in range(len(qa_docs))]

    embeddings = embedding.embed_documents(qa_docs)

    # add to collection
    collection.add(
        documents=qa_docs,
        embeddings=embeddings,
        ids=ids,
        metadatas=[{"question": q, "answer": a, "source": "AOU Oman FAQ v1"} for q, a in zip(questions, answers)]
    )

    print(f"Inserted {len(qa_docs)} Q/A pairs into collection '{collection_name}' with ids {ids[:3]}...")

    # testing
    query = "How may I apply to AOU?"
    query_embedding = embedding.embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    for rank, (doc, meta, id_) in enumerate(zip(results["documents"][0], results["metadatas"][0], results["ids"][0]),
                                            start=1):
        print(f"\nResult {rank} (id={id_}):")
        print(f"Question: {meta.get('question')}")
        print(f"Answer:   {meta.get('answer')}")
        print(f"Doc:      {doc}")


if __name__ == "__main__":
    print("Creating vector db")
    create_vector_db()
