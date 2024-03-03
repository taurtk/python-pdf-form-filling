
# Import necessary modules
import openai
import sys
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFDirectoryLoader

# Set up OpenAI API key
openai.api_key = ''

# Load PDF files from a directory
loader = PyPDFDirectoryLoader("./")
data = loader.load()
print(data)

# Split the extracted data into text chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)
text_chunks = text_splitter.split_documents(data)
print(len(text_chunks))

# Download the embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create embeddings for each text chunk
vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
persona ="""As a veteran seeking to apply for health benefits through the VA, you might find yourself overwhelmed by the complexity of the VA Form 10-10EZ. Imagine you're a seasoned veteran, navigating the bureaucratic maze of the VA health care system. You've heard that getting help is possible, and you're ready to dive into the form, section by section, to ensure your application is as accurate and complete as possible.

Let's break down the form into manageable parts, using a step-by-step approach to fill it out. Remember, each section is crucial for the VA to understand your eligibility and to provide the care you need.

General Information: Start with the basics. Fill in your name, address, and contact details. This is the foundation of your application.
Military Service Information: Dive into your military history. Detail your service branch, entry and discharge dates, and any disabilities or special conditions. This section is about your service and how it relates to your current health needs.
Insurance Information: Now, let's talk about your health insurance. List all your health insurance providers, including any coverage you have through a spouse or significant other. This section is about your health coverage and how it intersects with your VA benefits.
Dependent Information: If you have dependents, this is where you list them. Include your spouse and any children who are unmarried and under 18, or at least 18 but under 23 and attending school. This section is about your family and how they fit into your health care picture.
Employment Information: Next, update us on your employment status. If you're currently employed or retired, provide your employer's name, address, and contact information. This section is about your income and how it might affect your health care eligibility.
Financial Disclosure: Depending on your eligibility, you might need to disclose your financial information. Report your gross annual income and any other financial details relevant to your VA health care benefits. This section is about your financial situation and how it affects your health care needs.
Previous Calendar Year Gross Annual Income: Fill out this section with information about your gross annual income and that of your spouse and dependents. This is about your financial status over the past year.
Previous Calendar Year Deductible Expenses: Report any non-reimbursed medical expenses you or your dependents have paid. This is about your health care costs that weren't covered by insurance.
Consent to Copays and Communications: Finally, read through this section carefully and check the box to indicate your agreement. This is about your consent to pay copays and receive communications from the VA.
Review and Submit: Before submitting, review all the information you've entered. If you have any questions or need further assistance, don't hesitate to reach out.
By following these steps, you'll be able to navigate the VA Form 10-10EZ with confidence, ensuring your application is as complete and accurate as possible. Remember, every veteran's journey through the VA health care system is unique, and the VA is here to support you every step of the way."""

# Function to generate answers using OpenAI's API
def generate_answer(query, context):
    prompt = f"{persona} \n\n {context}\n\n{query} "
    completion = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {"role": "user", "content": prompt}
        ],
    )
    response = completion.choices[0].message.content
    return response

# Function to retrieve relevant text chunks and generate an answer
def ask_question(query):
    # Retrieve relevant text chunks
    k = 2 # Number of documents to retrieve
    retrieved_chunks = vector_store.search(query, search_type='similarity', k=k)
    print(retrieved_chunks)
    # Combine the retrieved chunks into a single context
    context = " ".join(str(item) for item in retrieved_chunks[0])
    
    # Generate an answer using the combined context
    answer = generate_answer(query, context)
    return answer

# Example usage
query = "What is the name"
answer = ask_question(query)
print(f"Answer: {answer}")

# Create an infinite loop to interact with the system
while True:
    user_input = input(f"Input Prompt: ")
    if user_input == 'exit':
        print('E  xiting')
        sys.exit()
    if user_input == '':
        continue
    # Pass the query to the system and print the response
    result = ask_question(user_input)
    print(f"Answer: {result}")
