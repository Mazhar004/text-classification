def get_model(model_path):
    with open(model_path + '/latest_model_path.txt', 'r') as fh:
        return(model_path+'/'+fh.readlines()[0])
