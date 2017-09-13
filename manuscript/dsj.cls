\LoadClass[]{article}

\RequirePackage{ifthen,array}
\RequirePackage{xspace}
\RequirePackage{everypage}
\RequirePackage{amsfonts}
\RequirePackage{amsthm}
\RequirePackage{amssymb}
\RequirePackage{geometry}
\RequirePackage{graphicx}
\RequirePackage{morefloats}
\RequirePackage{stfloats}
\RequirePackage{xcolor}
\RequirePackage{url}
\RequirePackage{txfonts}
\RequirePackage[authoryear]{natbib}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Page geometry
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\geometry{
 a4paper,
  paperwidth=6.85in,
  paperheight=9.72in,
  headheight=12pt,
  headsep=14pt,
  footskip=25pt,
  top=4pc,
  left=11.3pc,
  textwidth=26.85pc,
  bottom=5pc,
  footskip=2.2pc}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Font sizes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\renewcommand\normalsize{% Body of main text.
   \@setfontsize\normalsize{10}{12}%
   \abovedisplayskip \baselineskip%
   \abovedisplayshortskip .5\baselineskip%
   \belowdisplayskip \abovedisplayskip
   \belowdisplayshortskip \abovedisplayshortskip
   \let\@listi\@listI}
\normalsize
\renewcommand\small{%
   \@setfontsize\small{9}{10.8}%
   \abovedisplayskip 8.5\p@ \@plus3\p@ \@minus4\p@
   \abovedisplayshortskip \z@ \@plus3\p@
   \belowdisplayshortskip 5\p@ \@plus3\p@ \@minus3\p@
   \def\@listi{\leftmargin\leftmargini
               \topsep 6\p@ \@plus2\p@ \@minus2\p@
               \parsep 3\p@ \@plus2\p@ \@minus\p@
               \itemsep \parsep}%
   \belowdisplayskip \abovedisplayskip
}
\renewcommand\footnotesize{%
   \@setfontsize\footnotesize{8}{9.6}%
   \abovedisplayskip 6\p@ \@plus2\p@ \@minus4\p@
   \abovedisplayshortskip \z@ \@plus2\p@
   \belowdisplayshortskip 3\p@ \@plus2\p@ \@minus2\p@
   \def\@listi{\leftmargin\leftmargini
               \topsep 6\p@ \@plus2\p@ \@minus2\p@
               \parsep 3\p@ \@plus2\p@ \@minus\p@
               \itemsep \parsep}%
   \belowdisplayskip \abovedisplayskip
}

%% Colors
\def\color@dsjabs{xcolor@dsjabs}
\definecolor{xcolor@dsjabs}{cmyk}{.15,1,1,0}
\definecolor{dsjBlue}{cmyk}{1,0,0,0}
\def\color@dsjblue{dsjBlue}

\newcounter{dsj@tmpCtr}
\newcounter{dsj@auCtr}

\def\dsj@author{\stepcounter{dsj@auCtr}%
  \global\emailfalse\@ifnextchar[{\dsj@opt@author}{\dsj@norm@author}}

\def\dsj@opt@author[#1]#2{\expandafter\gdef\csname 
  dsj@@author\thedsj@auCtr\endcsname{\ignorespaces#2\textsuperscript{#1}}
  \setbox0=\hbox{#2}%
  \ifemail\gdef\t@corresp{#2}\fi%
  }

\def\dsj@norm@author#1{\expandafter\gdef\csname 
  dsj@@author\thedsj@auCtr\endcsname{#1}}

\let\author\dsj@author

\def\dsj@author@sep{%
  \ifnum\thedsj@tmpCtr=1\relax\else%
  \ifnum\count@dsj@auCtr=\thedsj@tmpCtr%
  \ifnum\count@dsj@auCtr=2\else\unskip,\fi\space%
  and\space\else\unskip,\space\fi\fi}

\def\dsj@printAuthors{\edef\count@dsj@auCtr{\thedsj@auCtr}%
 \setcounter{dsj@tmpCtr}{0}%
 \fontsize{10pt}{12}\selectfont%
 \whiledo{\count@dsj@auCtr>\thedsj@tmpCtr}
  {\stepcounter{dsj@tmpCtr}%
  \dsj@author@sep\csname dsj@@author\thedsj@tmpCtr\endcsname}%
  \normalsize}

% Address
\def\dsj@printAddress{}
\long\def\dsj@address{\@ifnextchar[{\dsj@opt@address}{\dsj@norm@address}}
\long\def\dsj@opt@address[#1]#2{\g@addto@macro\dsj@printAddress{\fontsize{9pt}{10pt}\selectfont%
  \par\leavevmode\llap{\textsuperscript{#1\,}}\textit{#2}}}
\long\def\dsj@norm@address#1{\g@addto@macro\dsj@printAddress{%
  \fontsize{8}{9.2}\selectfont\par\textit{#1}}}

\let\address\dsj@address

% Correspondence
\def\dsj@corresp{}
\def\corresp#1{\unskip\g@addto@macro\dsj@corresp{Correspondence to:
 #1}\ignorespaces}

\newif\ifemail
\let\@email\@empty
\def\email#1{\global\emailtrue\gdef\@email{#1}%
%  \xdef\@correspAuthor{\t@corresp}
  \immediate\write\@auxout{\string\gdef\string\num@correspau{\thedsj@auCtr}}%
%  \let\@correspAuthor\t@corresp
  }

%%% Ppage stlyes
% Title page
\def\ps@plain{
 \def\@oddhead{}
 \let\@evenhead\@oddhed
  \def\@oddfoot{\parbox[b]{\textwidth}{\color{\color@dsjblue}%
  \hfill\small\raise4pt\hbox{\thepage/\@totpages}}}
 }
\def\@totpages{0}

\AtEndDocument{\immediate\write\@auxout{\string\gdef\string\@totpages{\thepage}}}

% Other pages
\def\ps@otherpages{
  \gdef\@oddhead{%
  \llap{\fontsize{15pt}{15pt}%
  \color{\color@dsjabs}\sffamily\bfseries\selectfont%
  Design Science\hskip8pt}%
  \parbox[c]{\textwidth}{\color{\color@dsjabs}\hrule height .8pt width \textwidth}}
 \def\@evenfoot{%
 \parbox[b]{\textwidth}{\hfill\small\raise4pt\hbox{\thepage/\@totpages}}}
 \let\@oddfoot\@evenfoot}

\renewcommand\maketitle{\par
  \begingroup
    \def\@makefnmark{\rlap{\@textsuperscript{\normalfont\@thefnmark}}}%
    \long\def\@makefntext##1{\parindent 1em\noindent
            \hb@xt@1.8em{%
                \hss\@textsuperscript{\normalfont\@thefnmark}}##1}%
        \@maketitle
    \thispagestyle{plain}\@thanks
  \endgroup
  \setcounter{footnote}{0}%
%  \insert\footins{\footnotesize\dsj@corresp}
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@maketitle\relax
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\@title\@empty
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
  \pagestyle{otherpages}
}


%% Maketitle
\def\@maketitle{%
  \newpage
  \null
  \vspace*{-3.65em}%
  \begin{flushleft}%
  {\color{\color@dsjabs}%
  \fontsize{27pt}{27pt}\sffamily\bfseries\selectfont
  Design Science}\\[22.5pt]
  \let \footnote \thanks
    {\vspace{6.3\p@}\fontsize{21}{23}\sffamily\bfseries\boldmath\selectfont
    \@title \par}%
  \par\vspace{10pt}%
  \dsj@printAuthors
  \par\vspace{4pt}%
  \dsj@printAddress
  \par%
  \end{flushleft}%
  \par}

\let\@submitted\@empty
\def\submitted#1{\gdef\@submitted{#1}}

\AddEverypageHook{\setlength{\unitlength}{1pt}\thinlines%
\ifnum\thepage=1 
  \begin{picture}(0,0)(45,\strip@pt\textheight)%
\fboxsep 0pt%
\fboxrule 0pt%
\fbox{\parbox[b]{\dimexpr\pdfpagewidth-\textwidth-\Gm@rmargin-1.2pc-25pt}
  {\sffamily\fontsize{7pt}{8.2pt}\selectfont\raggedright%
  \def\xx@pc{8pt}%
  \ifx\@email\@empty\else%
   \textbf{\textcolor{\color@dsjblue}{Corresponding author}}\\
   \t@corresp\\
   \@email\\[\xx@pc]\fi
  Published by Cambridge\\
  University Press under\\
  the Creative Commons\\
  Attribution licence\\
  (\textcolor{\color@dsjblue}{http://creativecommons.org/}\\
  \textcolor{\color@dsjblue}{licenses/by/3.0/})
  \\[\xx@pc]
  \textcopyright\ The Author(s)\\[\xx@pc]
  \textbf{\textit{\textsf{Des. Sci.}}}\\
  journals.cambridge.org/dsj\\
  \textbf{\textcolor{\color@dsjblue}{DOI}}\\[\dimexpr(\xx@pc+4pt)]
  \includegraphics{dsj-logo.pdf}\\[\xx@pc]
  \includegraphics{cam-logo.pdf}
  }}\end{picture}\fi}

%%%% Abstract
\newbox\dsj@absbox
\renewenvironment{abstract}{%
  \global\setbox\dsj@absbox=\vbox\bgroup%
  \vspace{4pt}\fontsize{9pt}{10.5pt}\selectfont%
  \noindent{\fontsize{14pt}{14pt}%
  \selectfont\sffamily\textbf{\color{\color@dsjabs}\abstractname}}\\}{\egroup
  \dsj@printAbstract}

\def\dsj@printAbstract{
  \color{\color@dsjabs}\hrule height 1pt\normalcolor
  \par\vskip1pt
  \unvbox\dsj@absbox\par\vskip6pt
  \color{\color@dsjabs}\hrule height .5pt\normalcolor
  }

\def\dsj@printKeywords{}
\DeclareRobustCommand\keywords[1]{%
  {\par\vskip.5\baselineskip%
    \noindent{\small{\textbf{Keywords:}} #1}
     \par
     \vspace{\baselineskip}
    }}


%% Sectioning
\setcounter{secnumdepth}{0}
\def\@seccntformat#1{\csname the#1\endcsname.\hskip.7em}
\renewcommand\section{\@startsection {section}{1}{\z@}%
  {-3.5ex \@plus -1ex \@minus -.2ex}%
  {1ex \@plus.2ex}%
  {\color{\color@dsjabs}\fontsize{14pt}{1.8pt}\sffamily\selectfont\boldmath\bfseries}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
  {-3.25ex\@plus -1ex \@minus -.2ex}%
  {1ex \@plus .2ex}%
  {\normalfont\sffamily\fontsize{12pt}{14.4pt}\selectfont\boldmath\bfseries}}
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
  {-3.25ex\@plus -1ex \@minus -.2ex}%
  {.1ex \@plus .2ex}%
  {\normalfont\sffamily\fontsize{10pt}{12pt}\selectfont\boldmath
  \bfseries\itshape}}


\def\etal{\textit{et~al.}\xspace}

%% Running heads
\def\dsj@shorttitle{Missing short title!!}
\def\dsj@shortauthor{Missing short authors!!}
\def\shorttitle#1{\gdef\dsj@shorttitle{#1}}
\def\shortauthor#1{\gdef\dsj@shortauthor{#1}}
\let\@footer\@empty
\def\footer#1{\gdef\@footer{#1}}

%% Math and other parameters
\thinmuskip=3mu
\medmuskip=4mu plus 2mu minus 4mu
\thickmuskip=5mu plus 3mu minus 1mu
\def\defaultscriptratio{.65}
\def\defaultscriptscriptratio{.5}
\parindent=9pt
\parskip=0pt
\hyphenpenalty=9999
\tolerance=1000
\emergencystretch=5pt
\lineskiplimit=-5pt

% Full width figure
\newif\ifdsj@flt@full
\renewenvironment{figure*}[1][]
  {\@dblfloat{figure}[#1]
  \dsj@flt@fulltrue \leftskip -\dimexpr\Gm@lmargin-3.5pc}
  {\end@dblfloat}

\renewenvironment{table*}[1][]
  {\@dblfloat{table}[#1]
  \dsj@flt@fulltrue \leftskip -\dimexpr\Gm@lmargin-3.5pc
  \hsize=\dimexpr\textwidth+\Gm@lmargin-3.5pc}
  {\end@dblfloat}

% Caption
\abovecaptionskip = 5pt
\belowcaptionskip = 5pt
\def\xxx{\def\endtabular{\crcr\egroup\egroup $\egroup} %$
%\expandafter \let \csname endtabular*\endcsname = \endtabular
\def\tabular{\edef\@halignto{to\textwidth}\@tabular}
}

%\def\endtabular{\crcr\egroup\egroup $\egroup} %$
%\def\fo@r@arg#1#2#3#4#5\@nil{#1#2#3#4}

%\def\tabular{%
%\edef\@tempz{\expandafter\fo@r@arg\@currenvir\@nil}%
% \ifthenelse{\equal{\@tempz}{tabu}}{\let\@halignto\@empty}
% {\edef\@halignto{to\textwidth}}\@tabular}
\extrarowheight 3pt
\long\def\@makecaption#1#2{% 
  \ifdsj@flt@full \leftskip -\dimexpr\Gm@lmargin-3.5pc\fi
  \vskip\abovecaptionskip
  \small
  \sbox\@tempboxa{\textbf{\textsf{#1.}} #2}%
  \ifdim \wd\@tempboxa >\hsize
    \textbf{\textsf{#1.}} #2\par
  \else
    \global \@minipagefalse
    \hb@xt@\hsize{\ifdsj@flt@full\hskip -\dimexpr\Gm@lmargin-3.5pc\fi%
     \box\@tempboxa\hfill}%
  \fi
  \edef\@tempx{\expandafter\@carcube\@currenvir\@nil}%
  \ifthenelse{\equal{\@tempx}{tab}}{\aftergroup\small}{}%
  \vskip\belowcaptionskip}

% Float penalty
\setlength\floatsep    {1\baselineskip \@plus 2\p@ \@minus 2\p@}
\setlength\textfloatsep{1\baselineskip \@plus 2\p@ \@minus 4\p@}
\setlength\intextsep   {1\baselineskip \@plus 2\p@ \@minus 2\p@}

\renewenvironment{quote}
  {\list{}{\rightmargin\parindent\leftmargin\rightmargin\itshape}%
  \item\relax}
  {\endlist}

% Misc
\parindent .5cm
\setlength{\bibhang}{\parindent}
\def\etal{et~al.\xspace}
\urlstyle{rm}
\bibsep 3pt
\def\bibfont{\small\raggedright}
\arraycolsep 2pt

