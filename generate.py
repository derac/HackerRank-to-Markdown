import requests, argparse, os, json, shutil

from math import ceil
from time import sleep

parser = argparse.ArgumentParser()
parser.add_argument("-username","-u",required=True,type=str, help="HackerRank username")
parser.add_argument("-password","-p",required=True,type=str, help="HackerRank password")
parser.add_argument('-regenerate_cache','-regenerate','-r',action="store_true",help="Regenerate the cache.")
parser.add_argument("-pagination_limit","-l",default=300,type=int, help="Number of solution links to load per request. 500 is the highest I've tried that worked. ~1400 timed out.")
parser.add_argument("-timeout","-t",default=10,type=int, help="Number of seconds to wait between requests. I got banned briefly using someone else's tool. I'd err high.")
parser.add_argument("-useragent","-a",type=str, help="Useragent, you might run into a json error if it's out of date.",
                    default="'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36 Edg/85.0.564.63'")
args = parser.parse_args(); parser.print_help(); print();

run_path = os.path.dirname(os.path.realpath(__file__))
base_url = 'https://www.hackerrank.com/'
user_agent = args.useragent
login_url = base_url + 'auth/login'
submissions_url = base_url + 'rest/contests/master/submissions/?offset={}&limit={}'
challenge_url = base_url + 'rest/contests/master/challenges/{}/submissions/{}'
problem_url = base_url + 'challenges/{}/problem'
username, password = args.username, args.password
pages, timeout = args.pagination_limit, args.timeout

session = requests.Session() # log in
logon_response = session.post(login_url, auth=(username, password), headers={'user-agent': user_agent})
cookies, headers = session.cookies.get_dict(), logon_response.request.headers
logon_json = logon_response.json() # handle errors
if not logon_json["status"]: print(logon_json["errors"])

cache_path = os.path.join(run_path,"json_cache")
if args.regenerate_cache: shutil.rmtree(cache_path)
if not os.path.isdir(cache_path): os.mkdir(cache_path)

problem_list_path = os.path.join(cache_path,"problem_list") # Generate cache for the raw list of submissions
if not os.path.isdir(problem_list_path):
    os.mkdir(problem_list_path)
    total_results = session.get(submissions_url.format(0,0), headers=headers).json()["total"] # retrieving total results for pagination
    print("Total number of submissions:", total_results)

    for page in range(ceil(total_results/pages)):
        response = session.get(submissions_url.format(page*pages,pages), headers=headers)
        with open(os.path.join(problem_list_path,str(page)),"wb") as f: f.write(response.content)
        print("Page %s of %s written."%(page,pages)); sleep(timeout)

problems_dict = {} # Gather data for all problems in list
for page in os.listdir(problem_list_path):
    with open(os.path.join(problem_list_path,page),"rb") as f: problems_page = json.load(f)["models"]
    for problem in problems_page:
        if problem["challenge_id"] not in problems_dict:
            problems_dict[problem["challenge_id"]] = {"id": problem["id"],"slug": problem["challenge"]["slug"]}
print("Number of unique problems:", len(problems_dict))

problem_data_path = os.path.join(cache_path,"problem_data") # Generate cache for the raw problem data
if not os.path.isdir(problem_data_path): os.mkdir(problem_data_path)
for i, problem in enumerate(problems_dict.values()):
    problem_path = os.path.join(problem_data_path,str(problem["id"]))
    if not os.path.exists(problem_path):
        response = session.get(challenge_url.format(problem["slug"],problem["id"]), headers=headers)
        problem_json = response.json()
        if not problem_json["model"]: print("ERROR:",problem_json["message"]); break
        with open(problem_path,"wb") as f: f.write(response.content); sleep(timeout)
    print("Problem %s of %s written."%(i,len(problems_dict)))

problems_dict = {} # Gather data from all problems
for problem in os.listdir(problem_data_path):
    problem_path = os.path.join(problem_data_path,problem)
    with open(problem_path) as f: problem_json = json.loads(f.read())["model"]
    if problem_json["status"] == "Accepted":
        problems_dict[problem_json["id"]] = {item: problem_json[item] for item in
                                             ["language","code","name","slug","status","display_score"]}
print("Data from %s successful problems gathered."%len(problems_dict))

language_dict = {"javascript":"javascript","python3":"python","pypy3":"python",
                 "oracle":"sql","bash":"bash"}
with open(os.path.join(run_path,"README.md"),"w") as f:
    for problem in problems_dict.values():
        f.write("# [%s](%s) - %s - %s"%(problem["name"],problem_url.format(problem["slug"]),
                                        problem["language"],problem["display_score"]))
        language = language_dict[problem["language"]] if problem["language"] in language_dict else ''
        f.write("\n```%s\n%s\n```\n"%(language,problem["code"].strip()))
print("README.md written.")