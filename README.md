# Agentic HCMUT University Admission chatbot

Hi 👋 Welcome to the official repository for our **Agentic HCMUT University Admission Chatbot**!  

This project aims to build an intelligent, agentic chatbot to assist prospective students with information about Ho Chi Minh City University of Technology (HCMUT). The chatbot can answer questions regarding admission requirements, academic programs, campus facilities, application procedures, scholarships, and other university-related inquiries. By leveraging advanced NLP and RAG (Retrieval-Augmented Generation) technologies, our system provides accurate and helpful responses based on the university's official documentation and data.

**Features** ✨
- **Accurate Information Retrieval**: Answers questions using verified university data and documents
- **Vietnamese Language Support**: Optimized for both Vietnamese and English queries
- **Context-Aware Responses**: Maintains conversation context for more natural interactions
- **Semantic Search**: Uses advanced embedding techniques to understand question intent
- **Agentic Capabilities**: Can reason through complex multi-step questions about admissions

The system architecture is implemented based on paper [URAG: Implementing a Unified Hybrid RAG for Precise Answers in University Admission Chatbots – A Case Study at HCMUT](https://arxiv.org/pdf/2501.16276)

---

## Quick setup 🚀

The following steps will help you to get the system up and running:

- Create an `.env` file from template (adjust settings accordingly):
    ```bash
    make setup-env
    ```
- Setup Milvus Cloud for vector database:
    1. Visit [https://zilliz.com/cloud](https://zilliz.com/cloud) and create an account
    2. Create a new cluster
    3. Copy the cluster endpoint and token
    4. Fill the following variables in your `.env` file:
        - `MILVUS_CLOUD_URI`: Your cluster endpoint
        - `MILVUS_CLOUD_TOKEN`: Your cluster token
- Create network for the whole system. This will create network `chatbot` and add to `.env` file with the corresponding value of the network subnet:
    ```bash
    make create-network
    ```
- Build and start the server:
    ```bash
    make up-build-chatbot
    ```
- Upload data to the system by accessing [http://localhost:8003/docs](http://localhost:8003/docs):
    - Upload FAQ data in CSV format
    - Upload document data in TXT format
    - Or you can upload data from web URLs directly
- Access the chatbot interface at [http://localhost:8010](http://localhost:8010):

    ![Chatbot Interface](./media/web_ui.png)


---

We hope you enjoy exploring our project! If you have questions, feel free to open an issue or contribute to this repository. 😊 