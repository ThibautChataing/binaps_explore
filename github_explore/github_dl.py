import logging
import os
import datetime
from tracemalloc import start
import tqdm
import json
import argparse
import pandas as pd
import time
from enum import Enum
import sys
import traceback

from github import Github

class TqdmStream(object):
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
    # reate console handler for logger.
    ch = logging.StreamHandler(TqdmStream)
    ch.setLevel(level=logging.ERROR)
    ch.setFormatter(formatter)

    # add handlers to logger.
    if file_name:
        log.addHandler(fh)

    log.addHandler(ch)
    log.setLevel(level=logging.DEBUG)


    return log

class EventType(Enum):
    ONLYISSUE = 0
    ONLYPR = 1
    ISSUEPR = 2

class Event:
    repo: str
    id: int
    nbr: int
    etype: EventType
    participants: set

    def __init__(self, repo, id, nbr, etype=0) -> None:
        self.repo = repo
        self.id = id
        self.nbr = nbr
        self.etype = etype
        self.participants = set()

    def to_dataframe(self):
        return pd.DataFrame({'repo': self.repo, 'id': self.id, 'nbr': self.nbr, 'event_type': self.etype, 'participants': list(self.participants)})

    def to_list(self):
        return [self.repo, self.id, self.nbr, self.etype, self.participants]



def check_remaining_request(g):
    log = logging.getLogger('main')

    remaining = g.rate_limiting[0]
    if remaining < 4:
        log.critical(f'Time to sleep or change internet {remaining} requests')
        #i = input("Sleep or change IP ? 0/1")
        i = '0'
        if i == '0':
            end_sleep = datetime.datetime.fromtimestamp(g.rate_limiting_resettime)
            start_sleep = datetime.datetime.now()
            log.critical(f'Sleep from {start_sleep.strftime("%Y-%m-%dT%Hh%Mm%Ss")} to {end_sleep.strftime("%Y-%m-%dT%Hh%Mm%Ss")}')
            time.sleep(abs(int((end_sleep - start_sleep).total_seconds())) + 10)
        elif i == '1':
            check_remaining_request(g)
        else:
            pass
        log.critical('Waiting done, go again')

def get_from_named_user(named_user):
    return (named_user.id, named_user.login)

def get_event_from_pr(pr, repo, g):
    check_remaining_request(g)
    if pr.as_issue():
        typ = EventType.ISSUEPR.value
    else:
        typ = EventType.ONLYPR.value
    ev  = Event(repo=repo, id=pr.id, nbr=pr.number, etype=typ)

    check_remaining_request(g)
    ev.participants.add(get_from_named_user(pr.user))

    assignes = tuple(get_from_named_user(user) for user in pr.assignees)
    if assignes:
        ev.participants.add(assignes)

    check_remaining_request(g)
    # comments
    if pr.comments:
        com = pr.get_comments()
        for c in com:
            ev.participants.add(get_from_named_user(c.user))

    check_remaining_request(g)
    if pr.review_comments:
        com = pr.get_review_comments()
        for c in com:
            ev.participants.add(get_from_named_user(c.user))

    check_remaining_request(g)
    com = pr.get_issue_comments()
    if com.totalCount:
        for c in com:
            ev.participants.add(get_from_named_user(c.user))

    return ev


def error_log(log, err, sys_stack, repo_missing_path, repo, type):
    trace = sys_stack[2].tb_lineno
    log.error(err, sys_stack[1], trace)
    with open(repo_missing_path, 'a+') as fd:
        fd.write(f"{repo}, {type}\n")

def main(cpr=None):
    now = datetime.datetime.now().strftime("%Y-%m-%dT%Hh%Mm%Ss")

    # Argument management
    parser = argparse.ArgumentParser(description='github scrapping')
    parser.add_argument('-o', '--output', required=True,
                        help='output dir')
    parser.add_argument('-t', '--token', required=False,
                        help='token for github')

    args = parser.parse_args(cpr)
    root = args.output

    #  Logging managment
    log_file = os.path.join(root, fr'github_dl_{now}.log')
    set_logger(log_file)
    log = logging.getLogger('main')
    log.critical("start")

    #  Define path for input/output file
    repo_todo = os.path.join(root, r'repos_name.txt')
    repo_missing_path =  os.path.join(root, r'repos_name_missing.txt')

    #  Load github repo name to reach for data
    log.info(f'starting from {repo_todo}')
    with open(repo_todo, 'r') as fd:
        repos = fd.read()
    repos = repos.split(' ')
    repos.sort()
    log.debug(f"{len(repos)} repos to do")

    ### Main process
    g = Github(login_or_token=args.token)  # init github conenction
    for repo in tqdm.tqdm(repos,desc="Repos", leave=True, position=0):  # iterate over all repos
        df = pd.DataFrame(columns=['repo', 'id', 'nbr', 'event_type', 'participants'])

        #  Connect to repo
        try:
            log.info(f'Doing repo {repo}')
            check_remaining_request(g)
            rep = g.get_repo(repo)

        except Exception as err:
            error_log(log, err, sys.exc_info(), repo_missing_path, repo, 'get_repo')
            continue

        # Get all PR from the repo
        try:
            log.debug('Take pull request')
            check_remaining_request(g)
            prs = rep.get_pulls(state='all')  # get all pr
            for pr in tqdm.tqdm(prs, total=prs.totalCount, desc="PR", leave=False, position=1):
                check_remaining_request(g)
                ev = get_event_from_pr(pr, repo, g)
                df = pd.concat([df, ev.to_dataframe()], ignore_index=True)
        except Exception as err:
            error_log(log, err, sys.exc_info(), repo_missing_path, repo, 'pr')

        # Get all issues from the repo
        try:
            log.debug('Take issues')
            check_remaining_request(g)
            issues = rep.get_issues(state='all')
            for iss in tqdm.tqdm(issues,desc="Issue", total=prs.totalCount, leave=False, position=1):

                pr = iss.pull_request
                ev = Event(repo=repo, id=iss.id, nbr=iss.number)

                if pr:
                    ev.etype = EventType.ISSUEPR.value
                    check_remaining_request(g)
                    pr = iss.as_pull_request()
                    if pr.id in df.id:
                        continue
                    else:
                        check_remaining_request(g)
                        evpr = get_event_from_pr(pr, repo, g)
                        ev.participants = evpr.participants
                    df = pd.concat([df, ev.to_dataframe()], ignore_index=True)

                else:
                    ev.etype = EventType.ONLYISSUE.value
                    check_remaining_request(g)
                    ev.participants.add(get_from_named_user(iss.user))
                    check_remaining_request(g)
                    assignes = tuple(get_from_named_user(user) for user in iss.assignees)
                    if assignes:
                        ev.participants.add(assignes)

                    # comments
                    check_remaining_request(g)
                    com = iss.get_comments()
                    for c in com:
                        check_remaining_request(g)
                        ev.participants.add(get_from_named_user(c.user))
                    df = pd.concat([df, ev.to_dataframe()], ignore_index=True)
        except Exception as err:
            error_log(log, err, sys.exc_info(), repo_missing_path, repo, 'issue')

        try:
            log.info(f'{repo} done, saving it')
            repo_name = repo.replace('\\', '_')
            repo_name = repo_name.replace('/', '_')
            df.to_json(os.path.join(root, f'save_{repo_name}.json'))
        except Exception as err:
            error_log(log, err, sys.exc_info(), repo_missing_path, repo, 'saving')

    log.info("end")

       
if __name__ == "__main__":
    #args = "-o .\output"
    main() #args.split(' '))

