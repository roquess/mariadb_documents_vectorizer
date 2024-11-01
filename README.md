# mariadb_documents_vectorizer
Simple document vectorizer using mariadb vector as store writting in rust.

## Mariadb Vector docker

    docker pull quay.io/mariadb-foundation/mariadb-devel:11.6-vector-preview

    docker run --name mariadb-vector -e MYSQL_ROOT_PASSWORD=testvector -p 3306:3306 -d quay.io/mariadb-foundation/mariadb-devel:11.6-vector-preview    

    docker stop mariadb-vector
    docker start mariadb-vector

    To connect :
        docker exec -it mariadb-vector mariadb -uroot -p

    Create database vectordb.

## Parsing document

    cargo run -- index --file ressources/test.txt

## Retrieve elements

    cargo run -- search --query "Search question"
