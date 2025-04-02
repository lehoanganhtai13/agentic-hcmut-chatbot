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

- Create network for the whole system. This will create network `chatbot` and create an `.env` file with the corresponding value of the network subnet:
    ```bash
    make create-network
    ```
- Setup folders for containers' volume:
    ```bash
    make setup-volumes-minio
    ```
    ```bash
    make setup-volumes-milvus
    ```
- Start database services:
    ```bash
    make up-minio
    ```
    ```bash
    make up-milvus
    ```

---

## Backup/Restore database 💾

1. **Backup database**:
    - **Minio** (the folder name should follow the date format `dd/mm/yy`):
        ```
        make backup-minio FOLDER=dd/mm/yy
        ```
    - **Milvus**:
        1. Install **Go** if you did not:
            ```bash
            make install-go
            ```
        2. Clone `milvus-backup` repository:
            ```bash
            make clone-milvus-backup
            ```
        3. Build Milvus backup tool:
            ```bash
            make build-milvus-backup
            ```
        4. Synchronize configuration from `.env` file to the [configs.yaml](./database/backup/milvus-backup/configs/backup.yaml) file:
            ```bash
            make update-backup-config
            ```
        5. Running backup (the folder name should follow the date format `dd/mm/yy`):
            ```bash
            make backup-milvus FOLDER=dd/mm/yy
            ```
2. **Restore database**:
    - **Minio** (the folder name should follow the date format `dd/mm/yy`):
        ```
        make restore-minio FOLDER=dd/mm/yy
        ```
    - **Milvus** (If you have already completed steps 1 to 4 in the `Backup` section, you can skip them):
        1. Install **Go** if you did not:
            ```bash
            make install-go
            ```
        2. Clone `milvus-backup` repository:
            ```bash
            make clone-milvus-backup
            ```
        3. Build Milvus backup tool:
            ```bash
            make build-milvus-backup
            ```
        4. Synchronize configuration from `.env` file to the [configs.yaml](./database/backup/milvus-backup/configs/backup.yaml) file:
            ```bash
            make update-backup-config
            ```
        5. Running backup (the folder name should follow the date format `dd/mm/yy`):
            ```bash
            make restore-milvus FOLDER=dd/mm/yy
            ```

---

We hope you enjoy exploring our project! If you have questions, feel free to open an issue or contribute to this repository. 😊 