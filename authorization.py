from atlassian import Confluence
from dotenv import load_dotenv
import os

load_dotenv()

API_TOKEN = os.getenv("API_TOKEN")
URL = os.getenv("URL")
CLOUD = os.getenv("CLOUD")

confluence = Confluence(url=URL, 
                          cloud=CLOUD,
                        token=API_TOKEN)


# Method to get all permissions and restrictions of a specific page
# TODO: This is a bottom up approach and wildly inefficient as you would call it for all page ids, doing a bunch of duplicate checks. Change to top-down approach?
def getRestrictionsMetadata(page_id):
    # initialize lists
    permissions_users = []
    permissions_groups = []
    restrictions_users = []
    restrictions_groups = []

    # fetch the space id the page is apart of and all the corresponding page permissions
    space_id = confluence.get_page_space(page_id)
    all_perms = confluence.get_all_space_permissions(space_id)

    # loop over all the page permissions and append all the allowed group names and user keys to the lists
    for i in range(len(all_perms)):
        for op_sub in all_perms[i].items():
            if "operation" in op_sub[0]:
                if "read" in op_sub[1].get('operationKey'):
                    perm = all_perms[i].get("subject")
                    if perm.get("type") == "group":
                        permissions_groups.append(perm.get("name"))
                    elif perm.get("type") == "user":
                        permissions_users.append(perm.get("userKey"))

    # loop over all the parent id's of the page and append all the restrictions for groups and users separately
    cur_id = page_id
    while confluence.get_parent_content_id(cur_id) != -1 and confluence.get_parent_content_id(cur_id) != None:
        cur_id = confluence.get_parent_content_id(cur_id)
        restr = confluence.get_all_restrictions_for_content(content_id=cur_id)
        for group in restr.get("read").get("restrictions").get("group").get("results"):
            restrictions_groups.append(group.get("name"))
        for user in restr.get("read").get("restrictions").get("user").get("results"):
            restrictions_users.append(user.get("userKey"))
    

    allowed_groups = []
    allowed_people = []
    # if there are no restrictions, the space permissions apply
    if len(restrictions_groups) == 0 and len(restrictions_users) == 0:
        allowed_groups = permissions_groups
        allowed_people = permissions_users
    # if there are restrictions, take the intersection of the permissions and restrictions TODO: check if this is how permissions work
    # TODO: potentially append "permitted_group union restriction_group" to solve ambiguity in groups.  
    else:
        allowed_groups = list(set(permissions_groups).intersection(restrictions_groups))
        allowed_people = list(set(permissions_users).intersection(restrictions_users))

    return (allowed_groups, allowed_people)


