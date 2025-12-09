from lambda_function import lambda_handler

event = {
    "url": "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
}

result = lambda_handler(event, None)
print(result)
