import glob
import os
import multiprocessing
import subprocess

def run_local(pwd, n_jobs = -1, mode='r'):
    if n_jobs==-1:n_jobs=multiprocessing.cpu_count()
    p = multiprocessing.Pool(n_jobs)

    jobs, skipped = [], 0
    for job in glob.glob(pwd+'/*/run_local.py'):
        if os.path.isfile(job.rsplit('/',1)[0]+'/results.csv') and mode=='r':
            skipped+=1
            continue
        else:
            jobs.append(job)

    print('jobs:',len(jobs),'done:',skipped)

    jobs = [i for i in zip(['python']*len(jobs),jobs)]

    p.map(subprocess.call,jobs)

    p.close()


def run_condor(out_path, n_jobs=-1):
    '''
    '''
    n = str((len(glob.glob(out_path+'/*/setting.dlz'))))

    # create condor file:
    with open(out_path+'/run_condor.cmd','w') as f:
        f.write('executable      = '+out_path+'execute.sh\n')
        f.write('output          = '+out_path+'$(Process)/tmp.out\n')
        f.write('error           = '+out_path+'$(Process)/tmp.err\n')
        f.write('log             = '+out_path+'tmp.log\n')
        f.write('arguments       = $(Process)\n')
        f.write('queue '+n+'\n')

    subprocess.call(['condor_submit',out_path+'/run_condor.cmd'])
