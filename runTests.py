#!/usr/bin/env python3
# coding: utf-8

# This code runs the experiments for the paper:
# Title: Minimisation of Spatial Models using Branching Bisimilarity
# Authors: Vincenzo Ciancia, Jan Friso Groote, Diego Latella, Mieke Massink and Erik De Vink
# FORMAL METHODS 2023 https://fm2023.isp.uni-luebeck.de/index.php/overall-program/#fmacceptedpapers

# The code can be run as a notebook using visual studio code python notebook mode, by the special comments.
# Actually the script has been exported from an ipynb source.

# NOTE: this code is being distributed to reproduce the experiments, but it has not yet been optimized for readability/code review. This will be done before publication.

# NOTE: all rights reserved on any file in this archive. Due to double blind submission, proper attribution cannot be made explicit here. Licensing matters will be cleaned up before publication.

# %% Python setup
import subprocess 
import time
import glob
import os
from pathlib import Path
import shutil
import pandas as pd
from PIL import Image
import resource
from ast import literal_eval
import math

converter_exe_rel="./tools/GraphLogicA_0.6_linux-x64/GraphLogicA"
converter_exe = Path(converter_exe_rel).absolute().as_posix()
graphlogica_exe = converter_exe
#minimizer_exe = shutil.which("ltsconvert")
minimizer_exe = "./tools/mCRL2-202106/ltsconvert"
voxlogica_exe = "./tools/VoxLogicA_1.0-experimental_linux-x64/VoxLogicA"
output="output"
shutil.rmtree(output,ignore_errors=True)
os.makedirs(output,exist_ok=True)
images = glob.glob("test-images/*.png")

# function to run a OS command and measure the runtime; obj must include a "label" field, indicating the case in the dataset, and a "args" array (command and arguments)
def run(obj,print_output=False):
    print(f'''label: {obj["label"]}''')
    print(f'''command: {' '.join(map(str,obj["args"]))}''')
    start = time.perf_counter()
    my_env = os.environ.copy()
    my_env["LD_LIBRARY_PATH"] = "./tools/mCRL2.202106" 
    result = subprocess.run(obj["args"],capture_output=True,text=True,env=my_env)    
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        result.check_returncode()
    else:
        if print_output:
            print(result.stdout)
        return { "delta": time.perf_counter() - start, "label": obj["label"],"output": result.stdout,"error": result.stderr, "return_code": result.returncode }

# function to create a dataframe out of a list of results of the run function.
def mk_df(results,delta_label):
    return pd.DataFrame(results).set_index("label").rename(columns={"delta": delta_label}).drop(columns=["output","error","return_code"])


# %% Convert images to .aut format for minimization

print("Converting images to .aut ...")

# Given an image, return an object suitable as first argument for the run function defined above
def converter(image):
    path = Path(image)
    label = path.with_suffix("").name
    o_path = Path(output)
    s_path = path.with_suffix(".aut").name
    d_path = o_path.joinpath(s_path)
    return { "args": [converter_exe,"--convert",path.as_posix(),d_path.as_posix()], "label": label }

converter_result = [ run(converter(image)) for image in images ]
converter_df = mk_df(converter_result,"conversionAndWrite")


# %% Minimze .aut files using ltsconvert

print("minimizing .aut files...")

# the following is needed for working with large models in ltsconvert
resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

# Given an image, return an object suitable as first argument for the run function defined above
def minimizer(image):
    path = Path(image)
    label = path.with_suffix("").name
    o_path = Path(output)
    s_path = path.with_suffix(".aut").name
    d_path = o_path.joinpath(s_path)
    m_path = o_path.joinpath(
        Path(path.with_suffix("").name + "_min").with_suffix(".aut"))
    return {"args": [minimizer_exe, "--timings", "-ebranching-bisim", d_path.as_posix(), m_path.as_posix()], "label": label}


out = [run(minimizer(image)) for image in images]

def f(x: str):
    try:
        return float(x)
    except:
        return False

myLabels = { "reachability check": "reachabilityCheck", "total": "mcrl2-int" }

for o in out:
    lines: str = o["error"].strip("- ").splitlines()
    # res = {
    #     x[0]: f(x[1]) for line in lines if (x := line.strip("' ").split(":")) if len(x) > 1 if f(x[1])
    # }
    for line in lines:
        x = line.strip("' ").split(":")
        if len(x) > 1:
            fl = f(x[1])
            l = x[0]
            if l in myLabels.keys():
                l = myLabels[l]
            if fl:
                o[l] = fl

minimizer_df = mk_df(out, "mcrl2")
minimizer_df


# %% Convert the minimized model back
print("Converting minimized .aut to .json ...")

# Given an image, return an object suitable as first argument for the run function defined above
def convertBack(image):
    path = Path(image)
    label = path.with_suffix("").name
    o_path = Path(output)
    s_path = path.with_suffix(".aut").name
    d_path = o_path.joinpath(s_path)
    m_path = o_path.joinpath(Path(path.with_suffix("").name + "_min").with_suffix(".aut"))
    j_path = o_path.joinpath(path.with_suffix(".json").name)
    return { "args": [converter_exe,"--convert",m_path.as_posix(),j_path.as_posix()], "label": label }
    #return (run(label,args))

backConverter_df = mk_df([ run(convertBack(image)) for image in images ],"convertBack")


# %% Convert without writing the file, so that we gather the minimization time without I/O

print("Computing pure conversion times, without I/O ...")
# Given an image, return an object suitable as first argument for the run function defined above
def fakeConverter(image):
    path = Path(image)
    label = path.with_suffix("").name
    o_path = Path(output)
    s_path = path.with_suffix(".fake.aut").name
    d_path = o_path.joinpath(s_path)
    return { "args": [converter_exe,"--convert",path.as_posix(),d_path.as_posix(),"--fakeconversion"], "label": label }
    #return (run(label,args))

fakeConverter_df = mk_df([ run(fakeConverter(image)) for image in images ],"conversion")


# %% Model checking on images using VoxLogicA

print("Model checking full models ...")

def colour(r, g, b, is_graph=False):
    if is_graph:
        return f'''ap("{r:02x}{g:02x}{b:02x}")'''
    else:
        return f'''(red(img) =. {r}) & (green(img) =. {g}) & (blue(img) =. {b})'''


def save(basename, output,form, is_graph):
    p = Path(basename)
    n = p.with_suffix("").name
    if is_graph:
        return f'''save "{output}/{n}_{form}.json" {form}'''
    else:
        return f'''save "{output}/{n}_{form}.png" {form}'''


def mazeSpecification(path, is_graph):
    return f'''
            load img = "{path}"    
            let w = {colour(255,255,255,is_graph)}
            let b = {colour(0,0,255,is_graph)}
            let g = {colour(0,255,0,is_graph)}
            
            let zeta(phi1,phi2) = phi1 | through(N phi1,phi2)
         
            let form1 = zeta(b,w) & zeta(g,w)
            let form2 = b & (!zeta(zeta(g,w),b))
            let form3 = b & (zeta(zeta(g,w),b))
            {save(path,output,"form1",is_graph)}
            {save(path,output,"form2",is_graph)}
            {save(path,output,"form3",is_graph)}
        '''

def monoSpecification(path, is_graph):
    return f'''
            load img="{path}"
            
            let y = {colour(255,255,0,is_graph)}
            let c = {colour(0,255,255,is_graph)}
            let g = {colour(0,255,0,is_graph)}
            let m = {colour(255,0,255,is_graph)}
            let r = {colour(255,0,0,is_graph)}
            let b = {colour(0,0,255,is_graph)}
            let gr = {colour(191,191,191,is_graph)}
            let lgr = {colour(127,127,127,is_graph)}
            let lb = {colour(100,150,255,is_graph)}
            let lg = {colour(0,200,150,is_graph)}
            let lm = {colour(200,50,100,is_graph)}
            let bl = {colour(0,0,0,is_graph)}
            let w = {colour(255,255,255,is_graph)}
            let o = {colour(200,100,0,is_graph)}
            
            let zeta(phi1,phi2) = phi1 | through(N phi1,phi2)
            let ZZ(phi1,phi2) = (!phi2) & zeta(phi2,phi1)
            
            let form1 = y ZZ c ZZ g ZZ m ZZ r ZZ b ZZ gr ZZ bl ZZ w ZZ gr ZZ bl ZZ w ZZ lgr ZZ lb ZZ lg ZZ lm ZZ o
            {save(path,output,"form1",is_graph)}
        '''


def findSpec(img : str,is_graph = False):
    specs = [ ["maze",mazeSpecification],["mono",monoSpecification] ]
    for (prefix,spec) in specs:
        if Path(img).name.startswith(prefix):
            return spec(img,is_graph)
    return None

# Given an image or a graph, a specification, and the "is_graph" flag
# return an object suitable as first argument for the run function defined above
def modelChecker(image, spec, is_graph=False):
    path = Path(image)
    if is_graph:
        suffix=".grql"
    else:
        suffix=".imgql"
    fname = Path(output).joinpath(path.with_suffix(suffix).name)    
    f = open(fname, "w")
    f.write(spec)
    f.close()
    if is_graph:
        exe = graphlogica_exe
    else:
        exe = voxlogica_exe
    return {"args": [exe, fname], "label": path.with_suffix("").name, "property": "maze"}

modelChecker_df = mk_df(
    [run(modelChecker(image,spec))
        for image in images if (spec:=findSpec(image)) if spec],
    "modelCheckingFull")


# %% Model Checking on the minimal graph using GraphLogicA

print("Model checking minimal models ...")
def graph(image):
    path = Path(image)
    o_path = Path(output)
    j_path = o_path.joinpath(path.with_suffix(".json").name)
    return(j_path)

modelCheckerMin_df = mk_df(
    [run(modelChecker(graph(image),spec,True))
        for image in images if (spec:=findSpec(graph(image),True)) if spec],
    "modelCheckingMin")

# %% Read automata statistics
print("Gathering statistics ...")
def autSize(image):
    path = Path(image)
    label = path.with_suffix("").name
    o_path = Path(output)
    s_path = path.with_suffix(".aut").name
    d_path = o_path.joinpath(s_path)
    m_path = o_path.joinpath(Path(path.with_suffix("").name + "_min").with_suffix(".aut"))
    first_line = ""
    first_line_min = ""
    with open(d_path,"r") as file:
        first_line = file.readline().lstrip("des ")
    
    with open(m_path,"r") as file:
        first_line_min = file.readline().lstrip("des ")

    autSize = float(os.path.getsize(d_path)) / 1024
    minSize = float(os.path.getsize(m_path)) / 1024

    t = literal_eval(first_line)
    tmin = literal_eval(first_line_min)

    return { "transitions": t[1], "statesMin": tmin[2] , "transitionsMin": tmin[1], "label": label, "autSize": autSize, "minSize": minSize }

autSize_df = pd.DataFrame([ autSize(image) for image in images]).set_index("label")

# %% Gather image sizes and produce the final table


def size(imgpath):    
    path = Path(imgpath)
    img = Image.open(imgpath)
    imgSize = float(os.path.getsize(imgpath)) / 1024
    return { "pixels": img.width * img.height, "label": path.with_suffix("").name, "imgSize": imgSize}

size_df = pd.DataFrame([ size(image) for image in images]).set_index("label")



df = size_df.join(autSize_df).join(fakeConverter_df).join(converter_df).join(minimizer_df).join(backConverter_df).join(modelChecker_df).join(modelCheckerMin_df)


# %% Present data as in the paper

def convert_size(size_bytes,binary=False):
    if binary:
        size_name = ("", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB")    
    else:
        size_name = ("", "K", "M", "G", "T", "P", "E", "Z", "Y")
   
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s:.02f} {size_name[i]}"

df["prefix"] = df.index.str.split("-").map(lambda x: x[0])
df.sort_values(by=["prefix","pixels"],inplace=True)
df["convWIO"] = df["conversionAndWrite"]
df["computation"] = df["conversion"] + df["reachabilityCheck"] + df["reduction"] + df["convertBack"] + df["modelCheckingMin"]
df["min_computation"] = df["reachabilityCheck"] + df["reduction"]
df["speedupMC"] = df["modelCheckingFull"] / df["modelCheckingMin"]
df["pixels"] = df["pixels"].apply(convert_size)
df["transitions"] = df["transitions"].apply(convert_size)
df["autSizeH"] = df["autSize"].multiply(1024).apply(lambda x : convert_size(x,True))
df["minSizeH"] = df["minSize"].multiply(1024).apply(convert_size)
df["minWIO"] = df["mcrl2-int"]
df["total"] = df["conversion"] + df["reachabilityCheck"] + df["reduction"] + df["convertBack"] + df["modelCheckingMin"]

interestingdf = {
    "conversion": ["time","conversion"],
    "convWIO": ["t. + IO","conversion"],
    "pixels": ["states","full model"],
    "transitions": ["transitions","full model"],
    "autSizeH": ["aut file size","full model"],
    "min_computation": ["time","minimal model"],
    "minWIO": ["t. + IO","minimal model"],
    "statesMin": ["states","minimal model"],
    "transitionsMin": ["trans.","minimal model"],    
    "convertBack": ["t. back","model checking"],
    "modelCheckingFull": ["full","model checking"],
    "modelCheckingMin": ["min","model checking"],
    "speedupMC": ["speedup","model checking"]#,    
}

df = df.filter(interestingdf.keys())

# %% Save all data (raw and massaged)

df.to_csv("rawdata.csv")

print("\n\n*** Computations done. ***\n\nRaw data is in 'rawdata.csv', contents:\n")
with open('rawdata.csv') as x: print(x.read())

df.to_csv("results-table.csv",float_format="%.02f")

print("\n\nMassaged data is in 'results-table.csv', contents:\n")
with open('results-table.csv') as x: print(x.read())

print("\n\nAll done")