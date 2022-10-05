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
    etype: EventType
    participants: set

    def __init__(self, repo, id, etype=0, participants=set()) -> None:
        self.repo = repo
        self.id = id
        self.etype = etype
        self.participants = participants

    def to_dataframe(self):
        return pd.DataFrame({'repo': self.repo, 'id': self.id, 'event_type': self.etype, 'participants': list(self.participants)})

    def to_list(self):
        return [self.repo, self.id, self.etype, self.participants]


class ReposStatus(Enum):
    NOT_STARTED = 0
    DOING = 1
    DONE = 2

def check_remaining_request(g):
    log = logging.getLogger('main')

    remaining = g.rate_limiting[0]
    if remaining < 4:
        log.critical(f'Time to sleep or change internet {remaining} requests')
        #i = input("Sleep or change IP ? 0/1")
        i = '0'
        if i == '0':
            start_sleep = datetime.datetime.now()
            end_sleep = start_sleep + datetime.timedelta(hours=1)
            log.critical(f'Sleep from {start_sleep.strftime("%Y-%m-%dT%Hh%Mm%Ss")} to {end_sleep.strftime("%Y-%m-%dT%Hh%Mm%Ss")}')
            time.sleep(60*60 + 1)
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
    ev  = Event(repo=repo, id=pr.id, etype=typ)

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


def main():
    now = datetime.datetime.now().strftime("%Y-%m-%dT%Hh%Mm%Ss")
    parser = argparse.ArgumentParser(description='github scrapping')
    parser.add_argument('-o', '--output', required=True,
                        help='output dir')
    parser.add_argument('-t', '--token', required=False,
                        help='token for github')

    args = parser.parse_args()
    root = args.output

    log_file = os.path.join(root, fr'github_dl_{now}.log')
    set_logger(log_file)
    log = logging.getLogger('main')
    log.critical("start")

    whole_repo = os.path.join(root, r'repos_name.txt')
    repo_missing_path =  os.path.join(root, r'repos_name_missing.txt')
    data_path =  os.path.join(root, r'github_scrap_data.json')

    #if os.path.isfile(repo_missing_path):
    #    repo_todo = repo_missing_path
        #try:
            #df = pd.read_json(data_path)
        #except:
            #pass
            #df = pd.DataFrame(columns=['repo', 'id', 'event_type', 'participants'])

        #log.info('Process starting again from current repo_name_missing')
    #else:
    repo_todo = whole_repo
        #df = pd.DataFrame(columns=['repo', 'id', 'event_type', 'participants'])
    log.info('starting from scratch')
    
    with open(repo_todo, 'r') as fd:
        repos = fd.read()
    repos = repos.split(' ')
    repos.sort()
    repos_missing = repos.copy()
    log.debug(f"{len(repos)} repos to do")

    g = Github(login_or_token=args.token)  # init github conenction
    for repo in tqdm.tqdm(repos,desc="Repos", leave=True, position=0):  # iterate over all repos
        try:
            log.info(f'Doing repo {repo}')
            check_remaining_request(g)
            rep = g.get_repo(repo)

        except Exception as e:
            log.error(e)
            with open(repo_missing_path, 'a+') as fd:
                fd.write(f"{repo}, get_repo")
            continue

        try:
            log.debug('Take pull request')
            check_remaining_request(g)
            prs = rep.get_pulls(state='all')  # get all pr
            for pr in tqdm.tqdm(prs, total=prs.totalCount, desc="PR", leave=False, position=1):
                check_remaining_request(g)
                ev = get_event_from_pr(pr, repo, g)
        except Exception as e:
            log.error(e)
            with open(repo_missing_path, 'a+') as fd:
                fd.write(f"{repo}, pr")

        try:
            log.debug('Take issues')
            check_remaining_request(g)
            issues = rep.get_issues(state='all')
            for iss in tqdm.tqdm(issues,desc="Issue", total=prs.totalCount, leave=False, position=1):
                ev  = Event(repo=repo, id=iss.id)

                pr = iss.pull_request
                if pr:
                    ev.etype = EventType.ISSUEPR.value
                    check_remaining_request(g)
                    pr = iss.as_pull_request()
                    if pr.id in ev.id:
                        continue
                    else:
                        check_remaining_request(g)
                        evpr = get_event_from_pr(pr, repo, g)
                        ev.participants = evpr.participants

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
        except Exception as e:
            log.error(e)
            with open(repo_missing_path, 'a+') as fd:
                fd.write(f"{repo}, issue")

        try:
            log.info(f'{repo} done, saving it')
            repo_name = repo.replace('\\', '_')
            repo_name = repo_name.replace('/', '_')
            ev.to_dataframe().to_json(os.path.join(root, f'save_{repo_name}.json'))
        except Exception as e:
            log.error(e)
            with open(repo_missing_path, 'a+') as fd:
                fd.write(f"{repo}, saving")

    log.info("end")

       
if __name__ == "__main__":
    main()
