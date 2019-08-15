from datetime import datetime
import pickle
import logging.config

logger = logging.getLogger(__name__)


def save_to_pickle(file_name, data):
    path = "Pickles/" + get_file_name_datetime(file_name)
    pickle.dump(data, open(path, 'wb'))
    logger.info("{} saved to {}".format(file_name, path))


def retrieve_from_pickle(path, file_name='unknown'):
    try:
        data = pickle.load(path)
    except Exception:
        logger.exception(Exception)
        raise FileNotFoundError
    else:
        logger.info("Retrieved {} pickle from {} ".format(file_name, path))
        return data

def get_file_name_datetime(prefix, suffix='', file_type='p'):
    return prefix + "-" + str(datetime.now().strftime("%Y-%m-%d-%H%M")) + suffix + "." + file_type


