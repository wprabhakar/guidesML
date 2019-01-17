LOCAL_IP=`ipconfig getifaddr en0`
docker run -it --link d_mysql:mysql --rm mysql sh -c "exec mysql -h$LOCAL_IP -P3306 -uroot -ppassword"

