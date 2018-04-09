def update_log_file(exp_name,text,epoch_flage=False):
    
    def update(line,text,epoch_flage):
        if epoch_flage:
                line = line.split("_")[0]+"_"+text
        else:
                line = line = line.split(",")[0]+","+text
        return line
    try:
        lines = open("log.txt").read().split('\n')
        lines = [line if exp_name != line.split(",")[0] else update(line,text,epoch_flage) for line in lines]
        file = open('log.txt','w')
        file.write('\n'.join(lines))
        file.close()
        return True
    except:
        return False

