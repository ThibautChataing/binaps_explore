import json
import os
import sys
import logging
import tqdm
import datetime
import argparse
import requests
import time

class TqdmStream(object):
    """
    Streamer to properly display logging message during a tqdm display
    """
    @classmethod
    def write(_, msg):
        tqdm.tqdm.write(msg, end='')

    @classmethod
    def flush(_):
        sys.stdout.flush()

def set_logger(file_name=None, log_level=logging.DEBUG):
    # create logger for prd_ci
    """
    Args:
        file_name: name for the file to log (full path and extension ex: output/run.log)
        :param log_level:
    """
    log = logging.getLogger('main')
    log.propagate = False
    log.setLevel(level=logging.DEBUG)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('[%(levelname)s] (%(asctime)s) at %(lineno)dl : %(message)s')

    if file_name:
        # create file handler for logger.
        fh = logging.FileHandler(file_name, mode='w+')
        fh.setLevel(level=logging.DEBUG)
        fh.setFormatter(formatter)
    # create console handler for logger.
    ch = logging.StreamHandler(TqdmStream)
    ch.setLevel(level=logging.ERROR)
    ch.setFormatter(formatter)

    # add handlers to logger.
    if file_name:
        log.addHandler(fh)

    log.addHandler(ch)
    log.setLevel(level=logging.DEBUG)


    return log


def get_repo_descriptions(name, token, root):
    """ Download repository descriptions"""

    response = get_github_api(url=f'https://api.github.com/repos/{name}', token=token)

    save_to_json(response.json(), suffix='description', name=name, root=root)


def get_github_api(url, token, params=None):
    log = logging.getLogger('main')
    headers = {
        'Accept': 'application/vnd.github+json',
        'Authorization': f'Bearer {token}',
        'X-GitHub-Api-Version': '2022-11-28',
    }
    response = requests.get(url, headers=headers, params=params)

    remaining = response.headers['x-ratelimit-remaining']
    if int(remaining) < 2 :
        time = response.headers['x-ratelimit-reset']
        end_sleep = datetime.datetime.fromtimestamp(time)
        start_sleep = datetime.datetime.now()
        log.warning(f'Sleeping time : {remaining} remaining requests, sleep from {start_sleep.strftime("%Y-%m-%dT%Hh%Mm%Ss")} to {end_sleep.strftime("%Y-%m-%dT%Hh%Mm%Ss")}')
        duration = int((end_sleep - start_sleep).total_seconds()) + 10
        if duration < 0 :
            log.critical('Problems with sleeping duration')
            time.sleep(360)
        else:
            time.sleep(duration)

        log.warning('Waiting done, go again')
    
    return response


def get_save_PR(name, token, root):
    """Download information about all pull requests"""
    log = logging.getLogger('main')
    log.debug(f"Save PR for {name}")
    page = 1
    params = {
        'per_page': 100,
        'page': page

        }
    response = get_github_api(url=f'https://api.github.com/repos/{name}/pulls', token=token, params=params)

    save_to_json(response.json(), f'pullRequest{page}', name, root)

    while response.status_code == 200:
        get_commits(response.json(), name, token, root)
        
        page += 1
        params = {
            'per_page': 100,
            'page': page,
            'state': 'all'

        }
        response = get_github_api(url=f'https://api.github.com/repos/{name}/pulls', token=token, params=params)

        save_to_json(response.json(), f'pullRequest{page}', name, root)


def get_save_issue(name, token, root):
    """Download information about all pull requests"""
    log = logging.getLogger('main')
    log.debug(f"Save issue for {name}")
    page = 1
    params = {
        'per_page': 100,
        'page': page

        }
    response = get_github_api(url=f'https://api.github.com/repos/{name}/issues', token=token, params=params)

    save_to_json(response.json(), f'pullRequest{page}', name, root)

    while response.status_code == 200:
        page += 1
        params = {
            'per_page': 100,
            'page': page,
            'state': 'all'

        }
        response = get_github_api(url=f'https://api.github.com/repos/{name}/issues', token=token, params=params)
        save_to_json(response.json(), f'pullRequest{page}', name, root)


def get_save_issue_comments(name, token, root):
    """Download information about all comments in issue"""
    log = logging.getLogger('main')
    log.debug(f"Save issue comments for {name}")
    page = 1
    params = {
        'per_page': 100,
        'page': page

        }
    response = get_github_api(url=f'https://api.github.com/repos/{name}/issues/comments', token=token, params=params)

    save_to_json(response.json(), f'issueComment{page}', name, root)

    while response.status_code == 200:
        page += 1
        params = {
            'per_page': 100,
            'page': page,
            'state': 'all'

        }
        response = get_github_api(url=f'https://api.github.com/repos/{name}/issues/comments', token=token, params=params)
        save_to_json(response.json(), f'issueComment{page}', name, root)


def get_save_review_comments(name, token, root):
    """Download information about all comments in issue"""
    log = logging.getLogger('main')
    log.debug(f"Save review comments for {name}")
    page = 1
    params = {
        'per_page': 100,
        'page': page

        }
    response = get_github_api(url=f'https://api.github.com/repos/{name}/pulls/comments', token=token, params=params)

    save_to_json(response.json(), f'reviewComment{page}', name, root)

    while response.status_code == 200:
        page += 1
        params = {
            'per_page': 100,
            'page': page,
            'state': 'all'

        }
        response = get_github_api(url=f'https://api.github.com/repos/{name}/pulls/comments', token=token, params=params)
        save_to_json(response.json(), f'reviewComment{page}', name, root)


def get_commits(prs, name, token, root):
    """Get commits by PR"""
    log = logging.getLogger('main')
    log.debug(f"Save commit for {name}")
    for pr in tqdm.tqdm(prs, total=len(prs)):
        nbr = pr['number']

        page = 1
        params = {
            'per_page': 100,
            'page': page

            }
        response = get_github_api(url=f'https://api.github.com/repos/{name}/pulls/{nbr}/commits', token=token, params=params)

        save_to_json(response.json(), f'PR{nbr}_commits{page}', name, root)

        while response.status_code == 200:
            page += 1
            params = {
                'per_page': 100,
                'page': page,
                'state': 'all'

            }
            response = get_github_api(url=f'https://api.github.com/repos/{name}/pulls/{nbr}/commits', token=token, params=params)
            save_to_json(response.json(), f'PR{nbr}_commits{page}', name, root)


def save_to_json(data, suffix, name, root):
    log = logging.getLogger('main')
    name = name.replace('/', '_')
    file = os.path.join(root, f"{name}_{suffix}.json")
    with open(file, 'w') as fd:
        json.dump(data, fd)
    log.debug(f"File save at {file}")


def main(cpr=None):
    now = datetime.datetime.now().strftime("%Y-%m-%dT%Hh%Mm%Ss")

    # Argument management
    parser = argparse.ArgumentParser(description='github scrapping')
    parser.add_argument('-o', '--output', required=True,
                        help='dir for output files')
    parser.add_argument('-r', '--run_id', required=True,
                        help='run_id for db')
    parser.add_argument('-repo', '--repo_file', required=True, default='OSCP_data.db',
                        help='Source file with repos')
    parser.add_argument('-t', '--token', required=False,
                        help='token for github')
                        
    parser.add_argument('-ol', '--output_log', required=False, default=0,
                        help='dir for output log')

    args = parser.parse_args(cpr)
    if args.output_log :
        root_log = args.output_logs
    else:
        root_log = args.output

    #  Logging managment
    log_file = os.path.join(root_log, fr'github_dl_id{args.run_id}_{now}.log')
    set_logger(log_file)
    log = logging.getLogger('main')
    log.critical("start")

    # Get repo lists from a file that look like 1:['user/repo', 'other/truc'...]
    with open(args.repo_file, 'r') as fd:
        repos = json.load(fd)

    my_repo = repos[args.run_id]

    for rep in my_repo:
        get_repo_descriptions(rep, args.token, args.output)

        get_save_PR(rep, args.token, args.output)

        get_save_issue_comments(rep, args.token, args.output)

        get_save_review_comments(rep, args.token, args.output)


if __name__ == "__main__":
    args = r"-o C:\Users\Thibaut\Documents\These\code\binaps_explore\github_explore\test -r 1 -repo C:\Users\Thibaut\Documents\These\code\binaps_explore\github_explore\test\repo.json -t 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"
    main(args.split(' '))

