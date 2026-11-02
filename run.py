cd /home/administator/workspace/mp3
python3 server.py --port 8080

cd /home/administator/workspace/mp3 && nohup python3 server.py --host 0.0.0.0 --port 8000 > server.log 2>&1 &