import os
import pandas as pd
from collections import OrderedDict


'''
Create an -md.file with a table decribing e.g. .pdf-files in a figure folder.
'''


def init(md_filename="README", pkl_filename="info", path="/../output/figures/"):
    global MD_FILE, PKL_FILE, PATH_str
    MD_FILE = md_filename.replace(".md", "") + ".md"
    PKL_FILE = pkl_filename.replace(".pkl", "") + ".pkl"
    PATH_str = path


HERE = os.path.abspath(".")
try: 
    PATH = HERE + PATH_str
except NameError:
    init()
    PATH = HERE + PATH_str

deleteme = []
addme = []


def sayhello():
    try:
        infopd = pd.read_pickle(PATH + PKL_FILE)
    except FileNotFoundError:
        dummy = {"dummy.jpg": {"note": "delete me"}}
        dummypd = pd.DataFrame.from_dict(dummy, orient='index')
        dummypd.to_pickle(PATH + PKL_FILE)
        infopd = pd.read_pickle(PATH + PKL_FILE)

    global INFO, CATEGORIES
    INFO = infopd.transpose().to_dict()

    CATEGORIES = {}
    for fig in INFO.keys():
        for category in INFO[fig].keys():
            # FIXME
            CATEGORIES[category] = category
        break


def define_categories(categories:dict, include_note:bool=True):
    # Need to be called everytime, fix this?
    global CATEGORIES
    CATEGORIES = categories
    if not 'note' in CATEGORIES and include_note:
        CATEGORIES['note'] = 'note' #comment?





def set_file_info(filename, **params):
    INFO[filename] = {}
    for category in CATEGORIES.keys():
        try:
            s = str(params[category])
        except KeyError:
            s = None
        INFO[filename][CATEGORIES[category]] = s




def omit_category(category):
    # depricated (is this a word)
    infopd = pd.DataFrame.from_dict(INFO, orient='index')   
    infopd.pop(category)


def omit_file(filename):
    deleteme.append(filename)


def additional_information(bulletpoint):
    addme.append(bulletpoint)


def update(additional_information=addme, header=f"Description of plots in {PATH_str}"):
    info = OrderedDict(sorted(INFO.items(), key=lambda i:i[0].lower()))         # sort alphabetically
    infopd = pd.DataFrame.from_dict(info, orient='index')                       # create dataframe
    for filename in deleteme:                                                   # delete files in 'deleteme'-list
        infoT = infopd.transpose()
        try:
            infoT.pop(filename)
        except KeyError:
            print(f"There is no saved information about {filename} - please remove delete-command.")
        infopd = infoT.transpose()
    infopd.to_pickle(PATH + PKL_FILE)                                           # save in .pkl
    infopd.to_markdown(PATH + MD_FILE)                                          # create nice table in .md


    with open(PATH + MD_FILE, 'a') as file:
        file.write('\n\n\n')
        file.write(f'# {header}')
        file.write('\n\n')

        if len(additional_information)>0:
            file.write('\n## Additional information:\n\n')
            for line in additional_information:
                file.write(f'* {line}\n')

    print(f'\nSuccessfully written information to \n    {PATH_str}{MD_FILE}.\n')




#########
#Example#
#########

# import infoFile_ala_Nanna as Nanna
 
# Nanna.sayhello()
# Nanna.define_categories({'method':'method', 'opt':'optimiser', 'n_obs':r'$n_\mathrm{obs}$', 'no_epochs':'#epochs', 'eta':r'$\eta$', 'gamma':r'$\gamma$', 'rho':r'$\varrho_1$, $\varrho_2$'})

# Nanna.set_file_info("dummy1.jpg", note="helo", eta='0')
# #Nanna.omit_file("dummy1.jpg")
# #Nanna.omit_file("dummy2.jpg")
# # Nanna.set_file_info("dummy2.jpg", method='test', rho=1)

# Nanna.additional_information("hello")
# Nanna.update()

# def plot():
#     plt.plot()
    

#     if filename: 
#         plt.savefig(filname)

#         import infoFile_ala_Nanna as Nanna
#         Nanna.sayhello()
#         Nanna.define_categories({'method':'method', 'opt':'optimiser', 'n_obs':r'$n_\mathrm{obs}$', 'no_epochs':'#epochs', 'eta':r'$\eta$', 'gamma':r'$\gamma$', 'rho':r'$\varrho_1$, $\varrho_2$'})

#         Nanna.set_file_info(filename, note="helo", eta='0')


