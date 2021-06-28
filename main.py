import threading
import queue

pool = {}

def worker(worker_id):
    while True:
        item = pool[worker_id].get()
        print(f'Worker {worker_id} working on {item}')
        pool[worker_id].task_done()

def main():
    N = 3
    for i in range(N):
        pool[i] = queue.Queue()
        threading.Thread(target=worker,args=(i,), daemon=True).start()

    while True:
        job = int(input())
        pool[job].put('work')

if __name__ == '__main__':
    main()


