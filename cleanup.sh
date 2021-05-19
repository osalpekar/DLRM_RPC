kill $(ps -ef | grep python | grep osalpek | grep multiprocessing | awk '{print $2}')
