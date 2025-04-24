import requests
import csv
import time
import os.path
import pandas as pd
import os

github_con = ''
found = False

def get_github_token():
    token = os.getenv("GITHUB_TOKEN")
    if token is None:
        print("Environment variable 'GITHUB_TOKEN' not set.")
    return token

def check_rate_limit(token):
    headers = {'Authorization': f'token {token}'}
    response = requests.get('https://api.github.com/rate_limit', headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f'Rate limit check failed: {response.status_code} - {response.text}')
        return None

def search_github_repos(query, token, page):

    print('Page: ', page)

    headers = {'Authorization': f'token {token}'}
    base_url = 'https://api.github.com/search/repositories'

    search_query = f'page={page}&q=created:>2019-03-31+forks:>1+size:>150+{query}'
    search_url = f'{base_url}?{search_query}'

    try:
        response = requests.get(search_url, headers=headers)

        all_items = []
        total_count = 0

        if response.status_code == 200:
            current_items = response.json()['items'] 
            total_count = response.json()['total_count'] 
            
            all_items.extend(current_items)

            if total_count > 30 * page:
                time.sleep(2)
                next_items, _ = search_github_repos(query, token, page=page+1)
                all_items.extend(next_items)
                return all_items, total_count
        else:
            print(f'Error: {response.status_code} - {response.text}')
            return [], 0
    except:
        print('Error accessing GitHub API.')
        print('Sleeping for 30 seconds.')
        time.sleep(30)
        return [], 0

def find_file_in_repo(repo_full_name, filename_query, token):
    headers = {'Authorization': f'token {token}'}
    base_url = f'https://api.github.com/search/code'
    params = {
        'q': f'repo:{repo_full_name} {filename_query}'
    }
    
    response = requests.get(base_url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()['total_count'] > 0
    else:
        print(f'Error: {response.status_code} - {response.text}')
        return False

def check_config_folder_in_repo(repo, token):
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    api_url = "https://api.github.com/repos/"+repo
    contents_url = f"{api_url}/contents"

    print('Looking at: ', contents_url)

    response = requests.get(contents_url, headers=headers)

    if response.status_code == 200:
        endpoint = contents_url
        return get_contents(endpoint, headers, 1, "")
    else:
        print("Error: Unable to fetch repository data.")
        return False

def get_contents(endpoint, headers, deep, path):
    global found

    if deep <= 5: 
        print('Checking folder ', path)
        try:
            url = f"{endpoint}/{path}"
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                
                for entry in data:
                    if entry["type"] == "dir":
                        if entry["name"] == "config" or entry["name"] == "params" or entry["name"] == "launch" or entry["name"] == "bringup":
                            print('Folder ',entry["name"], ' found!')
                            found = True
                            return True
                        else:
                            if not found:
                                time.sleep(2)
                                get_contents(endpoint, headers, deep+1, entry["name"])
                    time.sleep(2)
            else:
                print("Failed to retrieve contents.")
                return False
        except:
            print('Error accessing GitHub API - get_content')
            print('Sleeping for 30 seconds.')
            time.sleep(30)
            return False

    return False

def save_to_csv(repos):
    with open('resources/repos.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Name', 'FullName', 'URL', 'Description', 'Size(KB)', 'IsFork', 'Created', 'Updated', 'Language', 'Forks', 'Topics', 'BranchesURL', 'CollaboratorsURL', 'Mined', 'Selected'])
        
        for repo in repos:
            writer.writerow([
                repo['name'], 
                repo['full_name'], 
                repo['html_url'], 
                repo['description'], 
                repo['size'],
                repo['fork'],
                repo['created_at'],
                repo['updated_at'],
                repo['language'],
                repo['forks'],
                repo['topics'],
                repo['branches_url'],
                repo['collaborators_url'],
                'no',
                'no'
            ])

def main():

    global found

    token = get_github_token()
    if not token:
        return

    print("Checking rate limit...")
    rate_limit_status = check_rate_limit(token)
    if rate_limit_status:
        remaining = rate_limit_status['rate']['remaining']
        print(f'API rate limit remaining: {remaining}')
        if remaining == 0:
            print('Rate limit exceeded. Please wait and try again later.')
            return

    if not os.path.exists('resources/repos.csv'):
        query = 'ros2'
        page = 1
        results = []
        
        try:
            print('Seeking repositories...', )
            repos, total_repos = search_github_repos(query, token, page)

            print('Repos found: ',total_repos)

            for repo in repos:
                repo_full_name = repo['full_name']
                results.append(repo)

            print(f"Total of projects selected due to size: {len(results)}")

            save_to_csv(results)

        except Exception as e:
            print(f"An error occurred: {e}")

    df = pd.read_csv('resources/repos.csv')

    filtered_df = df[df['Mined'] == 'no']

    for index in filtered_df.index:

        repo_full_name = df.at[index, 'FullName']
        selected = False

        print(f"Checking REQUIRED files in repo: {repo_full_name}...")
        found = False
        check_config_folder_in_repo(repo_full_name, token)
    
        if found:
            print('Selecting the repo ',repo_full_name)
        else:
            print('No config/launch found in ', repo_full_name)

        df.at[index, 'Mined'] = 'yes'

        if found:
            df.at[index, 'Selected'] = 'yes'

        df.to_csv('resources/repos.csv', index=False)

if __name__ == '__main__':
    main()
