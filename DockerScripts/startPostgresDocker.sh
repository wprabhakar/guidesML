docker run -d --name postgres -p 5432:5432 -e POSTGRES_DB=test_db -e POSTGRES_USER=test -e POSTGRES_PASSWORD=password -v /$(pwd)/:/var/lib/postgresql postgres:latest

