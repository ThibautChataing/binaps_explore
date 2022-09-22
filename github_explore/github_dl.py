import logging
import os
from unittest import skip
import tqdm
import pandas as pd
from enum import Enum

from github import Github

"""
def : 
    event = Issue alone / PR alone / PR from issue
Idea :
for each repo :
    ddl (event, participants)
        
"""

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

def get_from_named_user(named_user):
    return (named_user.id, named_user.login)

def get_event_from_pr(pr, repo):
    if pr.as_issue():
        typ = EventType.ISSUEPR.value
    else:
        typ = EventType.ONLYPR.value
    ev  = Event(repo=repo, id=pr.id, etype=typ)

    ev.participants.add(get_from_named_user(pr.user))

    assignes = tuple(get_from_named_user(user) for user in pr.assignees)
    if assignes:
        ev.participants.add(assignes)

    # comments
    if pr.comments:
        com = pr.get_comments()
        for c in com:
            ev.participants.add(get_from_named_user(c.user))
    if pr.review_comments:
        com = pr.get_review_comments()
        for c in com:
            ev.participants.add(get_from_named_user(c.user))


    com = pr.get_issue_comments()
    if com.totalCount:
        for c in com:
            ev.participants.add(get_from_named_user(c.user))

    return ev


def main():
    df = pd.DataFrame(columns=['repo', 'id', 'event_type', 'participants'])
    with open(r'C:\Users\Thibaut\Documents\These\code\binaps_explore\github_explore\repos_name.txt', 'r') as fd:
        repos = fd.read()
    repos = repos.split(' ')

    g = Github()  # init github conenction
    for repo in repos:  # iterate over all repos
        rep = g.get_repo(repo)
        """ prs = rep.get_pulls(state='all')  # get all pr
        for pr in prs:
            
            ev = get_event_from_pr(pr, repo)
            df = pd.concat([df, ev.to_dataframe()]) """

        issues = rep.get_issues(state='all')
        for iss in issues:
            ev  = Event(repo=repo, id=iss.id)

            pr = iss.pull_request
            if pr:
                ev.etype = EventType.ISSUEPR.value
                pr = iss.as_pull_request()
                if pr.id in df.id:
                    continue
                else:
                    evpr = get_event_from_pr(pr, repo)
                    ev.participants = evpr.participants

            else:
                ev.etype = EventType.ONLYISSUE.value
                
                ev.participants.add(get_from_named_user(iss.user))

                assignes = tuple(get_from_named_user(user) for user in iss.assignees)
                if assignes:
                    ev.participants.add(assignes)

                # comments
                com = iss.get_comments()
                for c in com:
                    ev.participants.add(get_from_named_user(c.user))


            df = pd.concat([df, ev.to_dataframe()])
            break
        break

if __name__ == "__main__":
    logging.basicConfig()

    logging.info("start")
    main()
    #g = Github()
    #rep = g.get_repo(repos[0])
    #pr = rep.get_pulls(state="all")
    #print(pr.totalCount)
    # for pr
    # pr.as_issue()
    # reviewer
    # assignees
    # comments  (get_comments / get_issue_comments)
    # user

    logging.info("end")

