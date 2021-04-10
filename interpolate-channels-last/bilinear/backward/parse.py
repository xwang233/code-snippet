import json


TIME_MULTIPLIER = 1e6
TIME_UNIT = 'us'

def main():
    with open('before.json', 'r') as f:
        j_bef = json.load(f)
    with open('after.json', 'r') as f:
        j_aft = json.load(f)
    
    with open('readme.md', 'w') as out_file:
        out_file.write(
            f'| shapes | cont({TIME_UNIT}) | cl_master({TIME_UNIT}) | cl_PR({TIME_UNIT}) | speed_up |\n')
        out_file.write('|' + '---|' * 5 + '\n')

        for key in j_bef:
            assert key in j_aft

            t_cont = j_bef[key]['cpu_time']
            t_cl_master = j_bef[key]['gpu_time']
            t_cl_pr = j_aft[key]['gpu_time']

            out_file.write(
                f'| {key} '
                f'| {t_cont :>.3f} '
                f'| {t_cl_master :>.3f} '
                f'| {t_cl_pr :>.3f} '
                f'| {t_cl_master / t_cl_pr :>.2f} | \n')



if __name__ == '__main__':
    main()