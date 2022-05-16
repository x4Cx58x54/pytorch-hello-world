from multiprocessing import Process

from config import configs
import train

def main():
    processes = []
    for conf in configs:
        processes.append(Process(target=train.main, args=(conf,)))
        print(conf)

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    print('done.')

if __name__ == '__main__':
    main()
