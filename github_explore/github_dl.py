import logging
import os
import datetime
import tqdm
import json
import pandas as pd
import time
from enum import Enum
import requests

from github import Github

def set_logger(file_name=None, log_level=logging.DEBUG):
    # create logger for prd_ci
    """
    Args:
        file_name: name for the file to log (full path and extension ex: output/run.log)
        :param log_level:
    """
    log = logging.getLogger('main')
    log.setLevel(level=logging.DEBUG)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('[%(levelname)s] %(module)s in %(funcName)s at %(lineno)dl : %(message)s')

    if file_name:
        # create file handler for logger.
        fh = logging.FileHandler(file_name, mode='w+')
        fh.setLevel(level=logging.DEBUG)
        fh.setFormatter(formatter)
    # reate console handler for logger.
    ch = logging.StreamHandler()
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
    repo : str
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

def check_remaining_request():
    log = logging.getLogger('main')

    r = requests.get('https://api.github.com/users/octocat')
    remaining = int(r.headers['X-RateLimit-Remaining'])
    log.debug(f"Still {remaining} requests")
    if remaining < 4:
        log.critical(f'Time to sleep or change internet {remaining} requests')
        #i = input("Sleep or change IP ? 0/1")
        i = '0'
        if i == '0':
            time.sleep(60*60+1)
        elif i == '1':
            check_remaining_request()
        else:
            pass
    logging.critical('Waiting done, go again')

def get_from_named_user(named_user):
    return (named_user.id, named_user.login)

def get_event_from_pr(pr, repo):
    check_remaining_request()
    if pr.as_issue():
        typ = EventType.ISSUEPR.value
    else:
        typ = EventType.ONLYPR.value
    ev  = Event(repo=repo, id=pr.id, etype=typ)

    check_remaining_request()
    ev.participants.add(get_from_named_user(pr.user))

    assignes = tuple(get_from_named_user(user) for user in pr.assignees)
    if assignes:
        ev.participants.add(assignes)

    check_remaining_request()
    # comments
    if pr.comments:
        com = pr.get_comments()
        for c in com:
            ev.participants.add(get_from_named_user(c.user))

    check_remaining_request()
    if pr.review_comments:
        com = pr.get_review_comments()
        for c in com:
            ev.participants.add(get_from_named_user(c.user))

    check_remaining_request()
    com = pr.get_issue_comments()
    if com.totalCount:
        for c in com:
            ev.participants.add(get_from_named_user(c.user))

    return ev


def main(root):
    log = logging.getLogger('main')

    whole_repo = os.path.join(root, r'repos_name.txt')
    repo_missing_path =  os.path.join(root, r'repos_name_missing.txt')
    data_path =  os.path.join(root, r'github_scrap_data.json')

    if os.path.isfile(repo_missing_path):
        repo_todo = repo_missing_path
        df = pd.read_json(data_path)
    else:
        repo_todo = whole_repo
        df = pd.DataFrame(columns=['repo', 'id', 'event_type', 'participants'])
    
    with open(repo_todo, 'r') as fd:
        repos = fd.read()
    repos = repos.split(' ')

    g = Github()  # init github conenction
    try:
        for repo in tqdm.tqdm(repos,desc="Repos", leave=True, position=2):  # iterate over all repos
            check_remaining_request()
            log.info(f'Doing repo {repo}')
            rep = g.get_repo(repo)

            check_remaining_request()
            prs = rep.get_pulls(state='all')  # get all pr
            for pr in tqdm.tqdm(prs, desc="PR", leave=True, position=3):
                check_remaining_request()
                ev = get_event_from_pr(pr, repo)
                df = pd.concat([df, ev.to_dataframe()])

            check_remaining_request()
            issues = rep.get_issues(state='all')
            for iss in tqdm.tqdm(issues,desc="Issue", leave=True, position=3):
                ev  = Event(repo=repo, id=iss.id)

                pr = iss.pull_request
                if pr:
                    ev.etype = EventType.ISSUEPR.value
                    check_remaining_request()
                    pr = iss.as_pull_request()
                    if pr.id in df.id:
                        continue
                    else:
                        check_remaining_request()
                        evpr = get_event_from_pr(pr, repo)
                        ev.participants = evpr.participants

                else:
                    ev.etype = EventType.ONLYISSUE.value
                    check_remaining_request()
                    ev.participants.add(get_from_named_user(iss.user))
                    check_remaining_request()
                    assignes = tuple(get_from_named_user(user) for user in iss.assignees)
                    if assignes:
                        ev.participants.add(assignes)

                    # comments
                    check_remaining_request()
                    com = iss.get_comments()
                    for c in com:
                        check_remaining_request()
                        ev.participants.add(get_from_named_user(c.user))
            log.info(f'{repo} done, saving it')
            df = pd.concat([df, ev.to_dataframe()])
            repo_name = repo.replace('\\', '_')
            ev.to_dataframe().to_json(os.path.join(root, f'save_{repo}.json'))

    except Exception as e:

            log.error(e)
            repos_missing = set(repos) - set(df.repo.unique()) 
            repos_missing.add(repo)
            log.info(f'saving repo todo and current data')
            with open(repo_missing_path, 'w') as fd:
                fd.write(' '.join(repos_missing))
            df.reset_index(inplace=True, drop=True)
            df.to_json(data_path)
            

if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%Hh%Mm%Ss")
    root = fr'C:\Users\Thibaut\Documents\These\code\binaps_explore\github_explore'
    log_file = os.path.join(root, fr'github_dl_{now}.log')
    set_logger(log_file)
    log = logging.getLogger('main')
    log.info("start")
    main(root)
    log.info("end")

