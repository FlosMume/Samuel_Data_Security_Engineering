import time
from flask import Flask, request

app = Flask(__name__)

@app.before_request
def start_timer():
    request.start_time = time.perf_counter()

@app.after_request
def log_request_details(response):
    process_time = time.perf_counter() - request.start_time
    print(f"Request: {request.method} {request.path} | Duration: {process_time:.4f}s | Status: {response.status_code}")
    return response

@app.route('/')
def hello():
    return 'Hello, World!'