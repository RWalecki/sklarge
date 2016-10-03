import glob
import os
import multiprocessing
import subprocess
from tqdm import tqdm

def run_local(pwd, n_jobs = 1, mode='r'):
    if n_jobs==-1:n_jobs=multiprocessing.cpu_count()
    p = multiprocessing.Pool(n_jobs)

    jobs, all, done = [], 0, 0
    for job in glob.glob(pwd+'/*/run_local.py'):
        all+=1
        if os.path.isfile(job.rsplit('/',1)[0]+'/results.csv') and mode=='r':
            done += 1
            continue
        jobs.append(job)

    print('jobs:',all)
    print('done:',done)

    if len(jobs)==0:return 0

    jobs = [i for i in zip(['python']*len(jobs),jobs)]
    if n_jobs==1:
        for j in tqdm(jobs):subprocess.call(j)
    else:
        p.map(subprocess.call,jobs)


def run_condor(pwd, n_jobs=-1, graphic_only=False):
    '''
    '''
    n = str((len(glob.glob(pwd+'/*/setting.dlz'))))
    if pwd[-1]!='/':pwd+='/'

    # create condor file:
    with open(pwd+'run_condor.cmd','w') as f:

        if graphic_only == True:
            f.write('letter          = substr(toLower(Target.Machine),0,7)\n')
            f.write('requirements    = stringListMember($(letter), "graphic")\n')
        else:
            f.write('letter          = substr(toLower(Target.Machine),0,7)\n')
            f.write('requirements    = !stringListMember($(letter), "graphic")\n')

        f.write('executable      = '+pwd+'execute.sh\n')
        f.write('output          = '+pwd+'$(Process)/tmp.out\n')
        f.write('error           = '+pwd+'$(Process)/tmp.err\n')
        f.write('log             = '+pwd+'tmp.log\n')
        f.write('arguments       = $(Process)\n')
        f.write('queue '+n+'\n')

    subprocess.call(['condor_submit',pwd+'run_condor.cmd'])
