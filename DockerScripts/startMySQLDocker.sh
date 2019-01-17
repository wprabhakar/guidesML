docker run -d --name=d_mysql -p 3306:3306 -e MYSQL_USER=test -e MYSQL_PASSWORD=password -e MYSQL_DATABASE=test_db -e MYSQL_ROOT_PASSWORD=password -v /$(pwd)/mysql-data:/var/lib/mysql mysql:latest
#--volume=/root/docker/test-mysql/conf.d:/etc/mysql/conf.d

