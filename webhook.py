import json

app = Flask(__name__)

@app.route('/',methods=['POST'])
def foo():
   data = json.loads(request.data)
   print "New commit by: {}".format(data['commits'][0]['H1']['name'])
   return "New central model file updated"

if __name__ == '__main__':
   app.run()
   
hospital1:~$ python sample.py 
 * Running on http://127.0.0.1:5000/
 
hospital1:~$ http POST http://127.0.0.1:5000 < sample.json
HTTP/1.0 200 OK
Content-Length: 2
Content-Type: text/html; charset=utf-8
Date: Sun, 27 July 2020 19:07:56 GMT
Server: Hospital/0.8.3 Python/3.6.5

OK # <-- this is the response the client gets

New commit by: Hospital 1
127.0.0.1 - - [27/July/2020 22:07:56] "POST / HTTP/1.1" 200 -
