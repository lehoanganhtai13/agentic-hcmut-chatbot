# Define variables
MINIO_DIR=./database/minio
MILVUS_DIR=./database/milvus
VALKEY_DIR=./database/valkey
MONGODB_DIR=./database/mongodb
FALKORDB_DIR=./database/falkordb
POSTGRES_DIR=./database/postgres
LANGFUSE_DIR=./observability/langfuse
CLICKHOUSE_DIR=./observability/clickhouse
WEB_CRAWLER_DIR=./web_crawler
CHATBOT_DIR=./chatbot
ENVIROMENT_DIR=./environments
BACKUP_DIR=./database/backup
MILVUS_BACKUP_CONFIG_DIR=./database/backup/milvus-backup/configs
SCRIPTS_DIR=./scripts
NETWORK_NAME=chatbot

# Define the default target
.PHONY: up-db down-db \
		create-network inspect-network remove-network \
		setup-volumes clean \
		backup restore \
		check-gpu

# Target to all services
up-minio:
	@echo "Starting Minio service..."
	@docker compose --env-file $(ENVIROMENT_DIR)/.env -f $(MINIO_DIR)/docker-compose.yml up -d

up-milvus:
	@echo "Starting Milvus service..."
	@docker compose --env-file $(ENVIROMENT_DIR)/.env -f $(MILVUS_DIR)/docker-compose.yml up -d

up-valkey:
	@echo "Starting Valkey service..."
	@docker compose --env-file $(ENVIROMENT_DIR)/.env -f $(VALKEY_DIR)/docker-compose.yml up -d

up-mongodb:
	@echo "Starting MongoDB service..."
	@docker compose --env-file $(ENVIROMENT_DIR)/.env -f $(MONGODB_DIR)/docker-compose.yml up -d

up-falkordb:
	@echo "Starting FalkorDB service..."
	@docker compose --env-file $(ENVIROMENT_DIR)/.env -f $(FALKORDB_DIR)/docker-compose.yml up -d

up-postgres:
	@echo "Starting Postgres service..."
	@docker compose --env-file $(ENVIROMENT_DIR)/.env -f $(POSTGRES_DIR)/docker-compose.yml up -d

up-clickhouse:
	@echo "Starting ClickHouse service..."
	@docker compose --env-file $(ENVIROMENT_DIR)/.env -f $(CLICKHOUSE_DIR)/docker-compose.yml up -d

up-langfuse:
	@echo "Starting Langfuse service..."
	@docker compose --env-file $(ENVIROMENT_DIR)/.env -f $(LANGFUSE_DIR)/docker-compose.yml up -d

up-build-chatbot:
	@echo "Building and starting Web Crawler service..."
	@docker compose -f $(WEB_CRAWLER_DIR)/docker-compose.yml up --build --wait -d
	@echo "Building and starting Chatbot service..."
	@docker compose --env-file $(ENVIROMENT_DIR)/.env -f $(CHATBOT_DIR)/docker-compose.yml up --build --wait -d

up-chatbot:
	@echo "Starting Web Crawler service..."
	@docker compose -f $(WEB_CRAWLER_DIR)/docker-compose.yml up --wait -d
	@echo "Starting Chatbot service..."
	@docker compose --env-file $(ENVIROMENT_DIR)/.env -f $(CHATBOT_DIR)/docker-compose.yml up --wait -d

up-db:
	@echo "Setting up database services..."
	@$(MAKE) up-minio
	@$(MAKE) up-milvus

down-minio:
	@echo "Stopping Minio service..."
	@docker compose --env-file $(ENVIROMENT_DIR)/.env -f $(MINIO_DIR)/docker-compose.yml down

down-milvus:
	@echo "Stopping Milvus service..."
	@docker compose --env-file $(ENVIROMENT_DIR)/.env -f $(MILVUS_DIR)/docker-compose.yml down

down-valkey:
	@echo "Stopping Valkey service..."
	@docker compose --env-file $(ENVIROMENT_DIR)/.env -f $(VALKEY_DIR)/docker-compose.yml down

down-mongodb:
	@echo "Stopping MongoDB service..."
	@docker compose --env-file $(ENVIROMENT_DIR)/.env -f $(MONGODB_DIR)/docker-compose.yml down

down-falkordb:
	@echo "Stopping FalkorDB service..."
	@docker compose --env-file $(ENVIROMENT_DIR)/.env -f $(FALKORDB_DIR)/docker-compose.yml down

down-postgres:
	@echo "Stopping Postgres service..."
	@docker compose --env-file $(ENVIROMENT_DIR)/.env -f $(POSTGRES_DIR)/docker-compose.yml down

down-clickhouse:
	@echo "Stopping ClickHouse service..."
	@docker compose --env-file $(ENVIROMENT_DIR)/.env -f $(CLICKHOUSE_DIR)/docker-compose.yml down

down-langfuse:
	@echo "Stopping Langfuse service..."
	@docker compose --env-file $(ENVIROMENT_DIR)/.env -f $(LANGFUSE_DIR)/docker-compose.yml down

down-crawler:

down-chatbot:
	@echo "Stopping Chatbot service..."
	@docker compose --env-file $(ENVIROMENT_DIR)/.env -f $(CHATBOT_DIR)/docker-compose.yml down
	@echo "Stopping Crawler service..."
	@docker compose -f $(WEB_CRAWLER_DIR)/docker-compose.yml down

down-db:
	@echo "Stopping database services..."
	@$(MAKE) down-milvus
	@$(MAKE) down-minio

# Targets for network
create-network:
	@echo "Creating network $(NETWORK_NAME)..."
	@docker network create $(NETWORK_NAME)
	@$(MAKE) update-env

inspect-network:
	@echo "Network subnet: $(shell docker network inspect $(NETWORK_NAME) | grep -o '"Subnet": "[^"]*' | sed 's/"Subnet": "//')"

update-env:
	@echo "Updating $(ENVIROMENT_DIR)/.env file with NETWORK_SUBNET..."
	@mkdir -p $(ENVIROMENT_DIR) && touch $(ENVIROMENT_DIR)/.env
	@SUBNET=$$(docker network inspect $(NETWORK_NAME) | grep -o '"Subnet": "[^"]*' | sed 's/"Subnet": "//') && \
	if grep -q '^NETWORK_SUBNET=' $(ENVIROMENT_DIR)/.env; then \
		sed -i '' "s~^NETWORK_SUBNET=.*~NETWORK_SUBNET=$$SUBNET~" $(ENVIROMENT_DIR)/.env; \
	else \
		echo "NETWORK_SUBNET=$$SUBNET" >> $(ENVIROMENT_DIR)/.env; \
	fi

remove-network:
	@echo "Removing network $(NETWORK_NAME)..."
	@docker network rm $(NETWORK_NAME)
	@echo "Removing NETWORK_SUBNET from $(ENVIROMENT_DIR)/.env..."
	@sed -i '' '/^NETWORK_SUBNET=/d' $(ENVIROMENT_DIR)/.env

# Target to create .env from template if it doesn't exist
setup-env:
	@echo "Checking for $(ENVIROMENT_DIR)/.env file..."
	@if [ ! -f "$(ENVIROMENT_DIR)/.env" ]; then \
		echo "$(ENVIROMENT_DIR)/.env not found. Creating from template..."; \
		cp "$(ENVIROMENT_DIR)/.template.env" "$(ENVIROMENT_DIR)/.env"; \
		echo "$(ENVIROMENT_DIR)/.env created successfully."; \
	else \
		echo "$(ENVIROMENT_DIR)/.env already exists. Skipping creation."; \
	fi

# Target to setup volumes folder
setup-volumes: 
	@$(MAKE) setup-volumes-minio
	@$(MAKE) setup-volumes-milvus
	@$(MAKE) setup-volumes-valkey
	@$(MAKE) setup-volumes-mongodb
	@$(MAKE) setup-volumes-falkordb

setup-volumes-minio:
	@echo "Creating volumes folder for minio..."
	@mkdir -p $(MINIO_DIR)/.data
	@echo "Setting permissions..."
	@sudo chmod -R 777 $(MINIO_DIR)/.data

setup-volumes-milvus:
	@echo "Creating volumes folder for milvus..."
	@mkdir -p $(MILVUS_DIR)/.data/etcd $(MILVUS_DIR)/.data/milvus
	@echo "Setting permissions..."
	@sudo chmod -R 777 $(MILVUS_DIR)/.data/etcd $(MILVUS_DIR)/.data/milvus

setup-volumes-valkey:
	@echo "Creating volumes folder for valkey..."
	@mkdir -p $(VALKEY_DIR)/.data
	@echo "Setting permissions..."
	@sudo chmod -R 777 $(VALKEY_DIR)/.data
	@sudo chmod 777 $(VALKEY_DIR)/valkey.conf

setup-volumes-mongodb:
	@echo "Creating volumes folder for mongodb..."
	@mkdir -p $(MONGODB_DIR)/.data $(MONGODB_DIR)/logs
	@echo "Setting permissions..."
	@sudo chmod -R 777 $(MONGODB_DIR)/.data $(MONGODB_DIR)/logs

setup-volumes-falkordb:
	@echo "Creating volumes folder for falkordb..."
	@mkdir -p $(FALKORDB_DIR)/.data
	@echo "Setting permissions..."
	@sudo chmod -R 777 $(FALKORDB_DIR)/.data
	@sudo chmod 777 $(FALKORDB_DIR)/falkordb.conf
	@sudo chmod 777 $(FALKORDB_DIR)/start.sh

setup-volumes-postgres:
	@echo "Creating volumes folder for postgres..."
	@mkdir -p $(POSTGRES_DIR)/.data
	@echo "Setting permissions..."
	@sudo chmod -R 777 $(POSTGRES_DIR)/.data

setup-volumes-clickhouse:
	@echo "Creating volumes folder for clickhouse..."
	@mkdir -p $(CLICKHOUSE_DIR)/.data/clickhouse_data $(CLICKHOUSE_DIR)/.data/clickhouse_logs
	@echo "Setting permissions..."
	@sudo chmod -R 777 $(CLICKHOUSE_DIR)/.data

setup-volumes-langfuse:
	@echo "Creating volumes folder for langfuse..."
	@mkdir -p $(MINIO_DIR)/.data/langfuse-data
	@echo "Setting permissions..."
	@sudo chmod -R 777 $(MINIO_DIR)/.data/langfuse-data

setup-volumes-chatbot:
	@echo "Creating volumes folder for chatbot..."
	@mkdir -p $(CHATBOT_DIR)/.data/logs $(CHATBOT_DIR)/.data/prometheus_multiproc
	@echo "Setting permissions..."
	@sudo chmod -R 777 $(CHATBOT_DIR)/.data/logs $(CHATBOT_DIR)/.data/prometheus_multiproc

# Target to clean up the volumes
clean:
	@$(MAKE) clean-milvus
	@$(MAKE) clean-minio
	@$(MAKE) clean-model
	@$(MAKE) clean-valkey
	@$(MAKE) clean-mongodb
	@$(MAKE) clean-falkordb

clean-milvus:
	@echo "Cleaning up Milvus volumes..."
	@docker run --rm -v $(MILVUS_DIR):/data alpine sh -c "rm -rf /data/.data"

clean-minio:
	@echo "Cleaning up Minio volumes..."
	@docker run --rm -v $(MINIO_DIR):/data alpine sh -c "rm -rf /data/.data"

clean-model:
	@echo "Cleaning up Model Serving volumes..."
	@docker run --rm -v $(MODEL_SERVING_DIR):/data alpine sh -c "rm -rf /data/.data"

clean-valkey:
	@echo "Cleaning up Valkey volumes..."
	@docker run --rm -v $(VALKEY_DIR):/data alpine sh -c "rm -rf /data/.data"

clean-mongodb:
	@echo "Cleaning up MongoDB volumes..."
	@docker run --rm -v $(MONGODB_DIR):/data alpine sh -c "rm -rf /data/.data"

clean-falkordb:
	@echo "Cleaning up Falkordb volumes..."
	@docker run --rm -v $(FALKORDB_DIR):/data alpine sh -c "rm -rf /data/.data"

clean-postgres:
	@echo "Cleaning up Postgres volumes..."
	@docker run --rm -v $(POSTGRES_DIR):/data alpine sh -c "rm -rf /data/.data"

clean-clickhouse:
	@echo "Cleaning up ClickHouse volumes..."
	@docker run --rm -v $(CLICKHOUSE_DIR):/data alpine sh -c "rm -rf /data/.data"

clean-langfuse:
	@echo "Cleaning up Langfuse volumes..."
	@docker run --rm -v $(MINIO_DIR):/data alpine sh -c "rm -rf /data/.data/langfuse-data"

clean-chatbot:
	@echo "Cleaning up Chatbot volumes..."
	@docker run --rm -v $(CHATBOT_DIR):/data alpine sh -c "rm -rf /data/.data/logs /data/.data/prometheus_multiproc"

# Target to backup the database
backup:
	@$(MAKE) backup-minio
	@$(MAKE) backup-milvus
	@$(MAKE) backup-mongodb
	@$(MAKE) backup-valkey
	@$(MAKE) backup-falkordb

# >>> make backup FOLDER=23-11-24

backup-minio:
	@echo "Backing up Minio data..."
	@mkdir -p $(BACKUP_DIR)/${FOLDER}/minio
	@sudo cp -r $(MINIO_DIR)/.data $(BACKUP_DIR)/$(FOLDER)/minio

backup-milvus:
	@echo "Backing up Milvus data..."
	@mkdir -p $(BACKUP_DIR)/${FOLDER}/milvus
	@$(BACKUP_DIR)/milvus-backup/milvus-backup create -n milvus_backup --config $(MILVUS_BACKUP_CONFIG_DIR)/backup.yaml
	@sudo cp -r $(MINIO_DIR)/.data/milvus-data-backup $(BACKUP_DIR)/$(FOLDER)/milvus

backup-mongodb:
	@echo "Backing up MongoDB data..."
	@mkdir -p $(BACKUP_DIR)/${FOLDER}/mongodb
	@sudo cp -r $(MONGODB_DIR)/.data $(BACKUP_DIR)/$(FOLDER)/mongodb
	@sudo cp -r $(MONGODB_DIR)/logs $(BACKUP_DIR)/$(FOLDER)/mongodb

backup-valkey:
	@echo "Backing up Valkey data..."
	@mkdir -p $(BACKUP_DIR)/${FOLDER}/valkey
	@sudo cp -r $(VALKEY_DIR)/.data $(BACKUP_DIR)/$(FOLDER)/valkey

backup-falkordb:
	@echo "Backing up FalkorDB data..."
	@mkdir -p $(BACKUP_DIR)/${FOLDER}/falkordb
	@sudo cp -r $(FALKORDB_DIR)/.data $(BACKUP_DIR)/$(FOLDER)/falkordb

# Target to restore the database
restore:
	@$(MAKE) restore-minio
	@$(MAKE) restore-milvus
	@$(MAKE) restore-mongodb
	@$(MAKE) restore-valkey
	@$(MAKE) restore-falkordb

# >>> make restore FOLDER=23-11-24

restore-minio:
	@echo "Restoring Minio data..."
	@sudo rsync -a --delete $(BACKUP_DIR)/${FOLDER}/minio/.data/ $(MINIO_DIR)/.data/
	@sudo chmod -R 777 $(MINIO_DIR)/.data

restore-milvus:
	@echo "Restoring Milvus data..."
	@sudo rsync -a --delete $(BACKUP_DIR)/${FOLDER}/milvus/milvus-data-backup/ $(MINIO_DIR)/.data/milvus-data-backup/
	@sudo chmod -R 777 $(MINIO_DIR)/.data
	@$(BACKUP_DIR)/milvus-backup/milvus-backup restore -n milvus_backup -s _recover --restore_index --config $(MILVUS_BACKUP_CONFIG_DIR)/backup.yaml

restore-mongodb:
	@echo "Restoring MongoDB data..."
	@sudo rsync -a --delete $(BACKUP_DIR)/${FOLDER}/mongodb/.data/ $(MONGODB_DIR)/.data/
	@sudo rsync -a --delete $(BACKUP_DIR)/${FOLDER}/mongodb/logs/ $(MONGODB_DIR)/logs/
	@sudo chmod -R 777 $(MONGODB_DIR)/.data $(MONGODB_DIR)/logs

restore-neo4j:
	@echo "Restoring Neo4j data..."
	@sudo rsync -a --delete $(BACKUP_DIR)/${FOLDER}/neo4j/.data/ $(NEO4J_DIR)/.data/
	@sudo chmod -R 777 $(NEO4J_DIR)/.data

restore-valkey:
	@echo "Restoring Valkey data..."
	@sudo rsync -a --delete $(BACKUP_DIR)/${FOLDER}/valkey/.data/ $(VALKEY_DIR)/.data/
	@sudo chmod -R 777 $(VALKEY_DIR)/.data

restore-falkordb:
	@echo "Restoring FalkorDB data..."
	@sudo rsync -a --delete $(BACKUP_DIR)/${FOLDER}/falkordb/.data/ $(FALKORDB_DIR)/.data/
	@sudo chmod -R 777 $(FALKORDB_DIR)/.data

# Target to check GPU
check-gpu:
	@echo "Checking GPU..."
	@docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi

# Install Go dependencies
install-go:
	@echo "Installing Go dependencies..."
	@chmod +x $(SCRIPTS_DIR)/install_go.sh
	@sudo $(SCRIPTS_DIR)/install_go.sh

# Clone Milvus backup repository
clone-milvus-backup:
	@echo "Cloning Milvus backup repository..."
	@git clone https://github.com/zilliztech/milvus-backup.git $(BACKUP_DIR)/milvus-backup

# Target to update Milvus backup configuration
update-backup-config:
	@echo "Updating backup.yaml with values from .env file..."
	@echo "Loading environment variables from $(ENVIROMENT_DIR)/.env..."
	@[ -f "$(ENVIROMENT_DIR)/.env" ] || { echo "Error: $(ENVIROMENT_DIR)/.env file not found"; exit 1; }

	@# Extract values from .env file
	@ACCESS_KEY_ID=$$(grep MINIO_ACCESS_KEY_ID $(ENVIROMENT_DIR)/.env | cut -d= -f2); \
	SECRET_ACCESS_KEY=$$(grep MINIO_SECRET_ACCESS_KEY $(ENVIROMENT_DIR)/.env | cut -d= -f2); \
	BUCKET_NAME=$$(grep MILVUS_MINIO_BUCKET_NAME $(ENVIROMENT_DIR)/.env | cut -d= -f2); \
	BACKUP_CONFIG=$(MILVUS_BACKUP_CONFIG_DIR)/backup.yaml; \
	\
	echo "Updating accessKeyID and backupAccessKeyID to $$ACCESS_KEY_ID..."; \
	sed -i '' 's/^  accessKeyID: .*$$/  accessKeyID: '"$$ACCESS_KEY_ID"'/' $$BACKUP_CONFIG; \
	sed -i '' 's/^  backupAccessKeyID: .*$$/  backupAccessKeyID: '"$$ACCESS_KEY_ID"'/' $$BACKUP_CONFIG; \
	\
	echo "Updating secretAccessKey and backupSecretAccessKey to $$SECRET_ACCESS_KEY..."; \
	sed -i '' 's/^  secretAccessKey: .*$$/  secretAccessKey: '"$$SECRET_ACCESS_KEY"'/' $$BACKUP_CONFIG; \
	sed -i '' 's/^  backupSecretAccessKey: .*$$/  backupSecretAccessKey: '"$$SECRET_ACCESS_KEY"'/' $$BACKUP_CONFIG; \
	\
	echo "Updating bucketName to $$BUCKET_NAME..."; \
	sed -i '' 's/^  bucketName: .*$$/  bucketName: "'"$$BUCKET_NAME"'"/' $$BACKUP_CONFIG; \
    \
    echo "Setting backupBucketName to milvus-data-backup..."; \
    sed -i '' 's/^  backupBucketName: .*$$/  backupBucketName: "milvus-data-backup"/' $$BACKUP_CONFIG; \

	@echo "Backup configuration updated successfully."

# Target to build Milvus backup tool
build-milvus-backup:
	@echo "Building Milvus backup tool..."
	@$(MAKE) -C $(BACKUP_DIR)/milvus-backup
	@echo "Milvus backup tool built successfully."

# Target to check connection before running backup
check-backup-connection:
	@echo "Checking connection to Minio..."
	@$(BACKUP_DIR)/milvus-backup/milvus-backup check --config $(MILVUS_BACKUP_CONFIG_DIR)/backup.yaml

# Target to create python requirements file
create-requirements:
	@echo "Creating requirements.txt file..."
	@pip list --format=freeze > $(CHATBOT_DIR)/server/requirements.txt
