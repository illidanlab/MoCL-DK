# Tempelate paper project for 2015Venue

Please read the [writing guidelines](https://github.com/illidanlab/labinfo/blob/master/lab_docs/writing.md) before you start to work on a new paper repository. Most of the content has been copied here, but may not be updated frequently.
Also, please modify the current file for introducing your repository.

### Repository

For each paper, create a **private** repository at [illidanlab](https://github.com/illidanlab).
Make sure you grant collaborators **permissions** for pushing. Using GitHub Desktop is 
recommended, to manage the commit, pull, push and etc.

The name of the repo should be `paper-(year)(venue)-(name)`. For example, a 
2016 ICDM paper about your algorithm named Epic, then the repo name should 
be `paper-2016ICDM-Epic`. 

Note: Github is offering unlimited private repos, please get the 
[Student Package](https://education.github.com/pack) for free from Github. 
If possible please use [two-factor authentication](https://github.com/blog/1614-two-factor-authentication) 
to protect our repos. 

### .gitignore file

Though one `.gitignore` file has been included for the minimal use case, it is still recommended to use the `gi` commandline tool to update the file with necessary ignore terms. For example,
```bash
# generate latex ignore content for macos platform and append to the end 
# of the `.gitignore` file.
$ gi macos,latex >> .gitignore
$ gi macos,python >> .gitignore
```

Exclude the main PDF file (e.g., main.tex or source/main.tex if the file is in a subfolder) as well. This has to be mannually done, since machines have no idea which file is your main pdf.

If you forgot to add the .gitignore file and Github is already tracking these intermediate files, you will need to first add the ignore file and also rebuilt the index, by using the following:
```bash
# Remove all cached files to ensure there are no .gitignore files being tracked
git rm --cached -r .

# Track the files that should be tracked
git add .
```

### File Structure

Use one tex file for each section, except for Conclusion. In the main file, have all the 
marco definitions, abstract, acknowledgement, conclusion, and include tex files for other 
sections. Example in the main file:
```tex
\section{Fancy Learning methods}
\label{sec:method}
\input{sec_method.tex}

\section{Experiments}
\label{sec:exp}
\input{sec_exp.tex}
```

In the section files e.g., `sec_exp.tex` and `sec_method.tex` in the previous example 
add one line in the very top of the file:
```tex
%!TEX root = main.tex
```
You should change `main.tex` to the main tex file in your project. You are encouraged 
to use `main.tex` so we know which file we should compile. 
Put all your figures to a sub-folder named `figures`. Whenever possible have all the 
figure source files (e.g., `.ppt` files) in the figures folder as well. The folder should 
contain only `tex` and `bib` files.

If you have to put some references in the repo, include all source files in a subfolder `source`. 
and create a new subfolder called  `references` and have all the files there. If you have raw experimental results 
in excel, use a subfolder called `experiment`. 

Repository 
+ [source](source)
  - [main.tex](source/main.tex): the main file 
  - [sec_intro.tex](source/sec_intro.tex)
  - [sec_method.tex](source/sec_method.tex)
  - [sec_exp.tex](source/sec_exp.tex)
  - [ref.bib](source/ref.bib)
  - [figures](source/figures)
    * [exp1.pdf](source/figures/demo.pdf)
+ [references](references)
  - [good_reference.pdf](references/good_reference.pdf)
+ [experiment](experiment)
  - [Mar27-Syn.md](experiment/Mar27-Syn.md) (or `.xlsx` file)

In this case, the `.gitignore` file should reflect the `main.tex` in a `source` folder. 

You should NOT include Acknowledgement in blind review, but you HAVE to add it 
when preparing the camera-ready. An example:
```tex
\section*{Acknowledgement}
This material is based in part upon work supported by the 
National Science Foundation under Grant IIS-1565596, IIS-1615597 and 
Office of Naval Research N00014-14-1-0631, N00014-17-1-2265. 
```
Pleaes check with me to confirm the grants to be acknowledged for your paper. 

### Editor + PDF Viewer

In general, use whatever you feel good: Sublime Text/Emacs/Vim. But sometimes you can 
gain a lot of efficiency. 

1. Mac OSX/Linux: Download [Sublime Text](https://www.sublimetext.com/3), install [Package Control](https://packagecontrol.io/installation) for Sublime Text, and install [LatexTools](https://github.com/SublimeText/LaTeXTools) and [Wrap Plus](https://github.com/ehuss/Sublime-Wrap-Plus) via Package Control.
Use [Skim](http://skim-app.sourceforge.net/) as the PDF reviewer, and configure Tex Sync in `Skim->Preferences->Sync`: activate `Check for file changes` and choose PDF-TeX Sync support to be `Sublime Text`. 
2. Windows: You are on your own. 

UPDATE: Both for Mac OS/Linux, you can use the [Visual Studio Code](https://code.visualstudio.com/) which contains more powerful tools for Latex and it is actively updated by Microsoft.co.

### Tex Engine
Use `xelatex`. Optionally, you may use `pdflatex`, if `xelatex` does not work for you. You 
can achieve by using a `%!TEX` directive:
```tex
% !TEX TS-program = xelatex
```
More details can be found [here](http://tex.stackexchange.com/questions/78101/when-and-why-should-i-use-tex-ts-program-and-tex-encoding).

### Writing

1. **Page Limits**. 
   1. When you are writing for a conference paper, ignore the page limits and write MORE than the limits. 
   It is much easier to trim down the content than to add additional contents in the last minute. 
   2. For the final submission version, use all the space allowed. If the page limit is 9, write 
   exactly 9 pages. You may have one spare line, but not more. No unnatural empty spaces aroung the figures and tables. 
   3. One more line beyond the required page limit will result your paper being rejected without review. 
2. **Goal of a paper**. The paper is to make people interested in and excited about your research and show clearly what you 
   have done. It is not a techical report for you to document only important details. 
3. **Get Familiar with Reviewers**. Whether or not your paper being accept or not is mainly controlled by the
   reviewers. So you should know what type of audience you have during the peer-review:
   1. Typically reviewers will spend 10 minutes to decide if your paper should be rejected or not. 
      How do reviewer spend this 10 minutes? Your abstract => introduction => How professional your paper 
      is (too many typo? grammar issues? page utilization? figures and tables look good? ). 
   2. Only after this 10 minutes he/she feel it worths his/her time to read more, the technical and other details will be read. 
   3. Don't make the assumption that the reviewers may know what you know. Write necessary details and cite key references along the way.
3. **Logics of a paragraph**. There should be logical connections among every two sentences next to each other. 
   Simply stacking sentences came out of your brain typically won't work. So, organize the sentences. 
4. **Figures**. Use vector based PDF for papers. 
   1. For MATLAB output, use `print` command and the `dpdf` option. See [here](http://www.mathworks.com/help/matlab/ref/print.html?refresh=true) for more details. Increase the size of the font in legends and axis, weight of lines, to make sure they are large enough when show in the paper. You should install Acrobat Professional 
      to crop the excess margin. 
   2. For concept figures/flow charts, use PowerPoint, select the components, and right click to save as PDF. The size 
      should be exactly what you wanted. 
   3. Using relative width to control size is preferred. e.g., `\includegraphics[width=0.9\textwidth]{figures/exp_syn.pdf}`. 
   4. Always use `\begin{figure}[t!]` to make it top unless otherwise necessary. Use commands such as `vspace{-0.1in}` to remove unnecessary blank spaces between the figure and caption. 
   5. In the caption include: the details about the figure, and what are the important findings in the figure. 
      1. Bad Example: The performance of A. 
      2. Good Example: Predictive performance on dataset D of the proposed method and competing methods. 
         The performance is measured by AUC. The average and variance on 10 random splittings are reported. 
         We see the proposed method outperforms the competing methods when data size is small. 
5. **Tables**. 
   1. Use `\small` if the table is too large. 
   2. Try to design different layouts to effective utilize the space, and avoid too much waste of space. 
   3. Always use `\begin{table}[t!]` to make it top unless otherwise necessary. Use commands such as `vspace{-0.1in}` to remove unnecessary blank spaces between the caption and tabular. 
   4. In the caption include: the details about the table, and what are the important findings in the table. 
   5. After you decide the layout, if you have many numbers to fill in that are generated by matlab, consider 
      writing a matlab script to output the latex code for the `tabular` environment. 
6. **Need Extra Space?**. 
   1. Use space control such as `\vspace{-0.1in}` to remove unnecessary space from figures and tables. 
   2. Remove `enumerate` or `itemize` environments. Use `\noindent\textbf{1. blah}` instead. 
   3. If the last line of a paragraph has only a few words (i.e., one or two), then try to 
      rephrase the last two to three sentences to reduce their lengths. 
   4. Use `\small` before `tabular` environments
   5. Use `\small` before bibliography (please read the submission guideline and see if this is allowed). 
   6. Resize and reorganize the figures into subfigures, and move them around the places. 
   7. Some math notations may take too much space, e.g., use `\sum\nolimits_a^b` instead of `\sum_a^b`. 
      Use `\tfrac{a}{b}` instead of `\frac{a}{b}`. Remove `\left` and `\right` for some brackets. 
   8. Ask me for help. 
7. **My Comments**.
   If you have a complete and nice draft one week before the deadline, you will have my extensive 
   comments in the form of PDF. And you will learn a lot. Otherwise there is no guarantee that 
   your paper will be ready by the time of submission, and you may be prohibited for submission 
   if the quality of the paper is poor. 

### Day of Submission

Revise your paper until the last minute. Common practice:

1. Submit once 3 hours before the deadline
2. Submit once 2 hours before the deadline
3. Submit once 1 hour before the deadline
4. Submit once half an hour before the deadline. 
5. Submit submit every time you find bugs thereafter.

CMT Systems *typically* open for an extra of 5- 10 minutes. You can continue 
find typos and do minor adjustments before the system is closed. But no major 
revisions should be done within the time window. 

### Post Submission

Once you submitted a paper. It is time for you to relax for a few days, and enjoy the life with 
friends and family. When you are relaxed, you gain energy to fight for another project/deadline. 
Forget about the submission, and don't think much if the paper is going to be accepted or rejected. 
Top conferences have an acceptance rate of 25% or less. A classifier that simply assumes all papers 
are rejected, gets a sound accuracy of 75% or more. 
