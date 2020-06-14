"""
\begin{table*}[tb]
\begin{center}
\caption{Results of Multi-instrument Source Separation}
\resizebox{0.7\textwidth}{!}{%
\LARGE
\begin{tabular}{|ccccccc}
\cline{1-6}
\multicolumn{1}{|c|}{Model} & \multicolumn{1}{c|}{Vocals} & \multicolumn{1}{c|}{Drums} & \multicolumn{1}{c|}{Bass} & \multicolumn{1}{c|}{Rest} & \multicolumn{1}{c|}{Overall} &  \\ \hline

\multicolumn{1}{|c|}{\multirow{3}{*}{\begin{tabular}[c]{@{}c@{}}Dedicated  \\ U-Nets (x4)\end{tabular}}} &
\multicolumn{1}{c|}{\sdrcolor \textbf{4.96} \(\pm\) 4.63 \textbf{(5.77)}} &
\multicolumn{1}{c|}{\sdrcolor 4.95 \(\pm\) 3.56 (4.60)} &
\multicolumn{1}{c|}{\sdrcolor \textbf{2.78} \(\pm\) 4.41 \textbf{(3.19)}} &
\multicolumn{1}{c|}{\sdrcolor 1.21 \(\pm\) 3.38 (2.23)} &
\multicolumn{1}{c|}{\sdrcolor 3.48 \(\pm\) 4.30 (3.61)} & \multicolumn{1}{c|}{\sdrcolor SDR} \\ \cline{7-7}
\multicolumn{1}{|c|}{} &
\multicolumn{1}{c|} {\sircolor 10.70 \(\pm\) 5.05 (11.19)} &
\multicolumn{1}{c|}{\sircolor 10.24 \(\pm\) 4.04 (9.66)} &
\multicolumn{1}{c|}{\sircolor5.83 \(\pm\) 5.37 (5.57)} &
\multicolumn{1}{c|}{\sircolor 5.56 \(\pm\) 3.35 (6.31)} &
\multicolumn{1}{c|}{\sircolor 8.08 \(\pm\) 5.09 (8.04)} & \multicolumn{1}{c|}{\sircolor SIR} \\ \cline{7-7}
\multicolumn{1}{|c|}{} &
\multicolumn{1}{c|} {\sarcolor \textbf{7.15} \(\pm\) 3.38 (7.20)} &
\multicolumn{1}{c|}{\sarcolor 7.26 \(\pm\) 3.23 (6.75)} &
\multicolumn{1}{c|}{\sarcolor 7.86 \(\pm\) 3.03 \textbf{(8.02)}} &
\multicolumn{1}{c|}{\sarcolor 4.73 \(\pm\) 2.83 (5.23)} &
\multicolumn{1}{c|}{\sarcolor 6.75 \(\pm\) 3.32 (6.67)} & \multicolumn{1}{c|}{\sarcolor SAR} \\ \hline \hline \hline

\multicolumn{1}{|c|}{\multirow{3}{*}{\begin{tabular}[c]{@{}c@{}}C-U-Net\end{tabular}}} &
\multicolumn{1}{c|}{\sdrcolor 4.49 \(\pm\) 4.75 (5.26)} &
\multicolumn{1}{c|}{\sdrcolor 4.54 \(\pm\) 3.59 (4.30)} &
\multicolumn{1}{c|}{\sdrcolor 2.51 \(\pm\) 4.26 (2.97)} &
\multicolumn{1}{c|}{\sdrcolor 0.97 \(\pm\) 3.57 (1.69)} &
\multicolumn{1}{c|}{\sdrcolor 3.13 \(\pm\) 4.31 (3.37)} & \multicolumn{1}{c|}{\sdrcolor SDR} \\ \cline{7-7}
\multicolumn{1}{|c|}{} &
\multicolumn{1}{c|} {\sircolor \textbf{11.33} \(\pm\) 4.80 \textbf{(11.97)}} &
\multicolumn{1}{c|}{\sircolor 10.80 \(\pm\) 4.15 \textbf{(10.77)}} &
\multicolumn{1}{c|}{\sircolor \textbf{6.60} \(\pm\) 4.91 \textbf{(6.40)}} &
\multicolumn{1}{c|}{\sircolor \textbf{6.12} \(\pm\) 3.20 (6.56)} &
\multicolumn{1}{c|}{\sircolor \textbf{8.71} \(\pm\) 4.90 \textbf{(8.79)}} & \multicolumn{1}{c|}{\sircolor SIR} \\ \cline{7-7}
\multicolumn{1}{|c|}{} &
\multicolumn{1}{c|} {\sarcolor 6.23 \(\pm\) 3.87 (6.77)} &
\multicolumn{1}{c|}{\sarcolor 6.49 \(\pm\) 3.34 (5.88)} &
\multicolumn{1}{c|}{\sarcolor 6.46 \(\pm\) 3.54 (6.57)} &
\multicolumn{1}{c|}{\sarcolor 3.98 \(\pm\) 3.35 (4.43)} &
\multicolumn{1}{c|}{\sarcolor 5.79 \(\pm\) 3.66 (5.81)} & \multicolumn{1}{c|}{\sarcolor SAR} \\ \hline \hline

\multicolumn{1}{|c|}{\multirow{3}{*}{\begin{tabular}[c]{@{}c@{}}UW\end{tabular}}} & \multicolumn{1}{c|}{\sdrcolor 4.31 \(\pm\) 4.80 (5.46)} &
\multicolumn{1}{c|}{\sdrcolor 5.19 \(\pm\) 3.51 (4.72)} &
\multicolumn{1}{c|}{\sdrcolor 2.55 \(\pm\) 4.58 (2.81)} &
\multicolumn{1}{c|}{\sdrcolor 1.51 \(\pm\) 3.32 (2.49)} &
\multicolumn{1}{c|}{\sdrcolor 3.39 \(\pm\) 4.32 (3.58)} & \multicolumn{1}{c|}{\sdrcolor SDR} \\ \cline{7-7}
\multicolumn{1}{|c|}{} &
\multicolumn{1}{c|} {\sircolor 9.95 \(\pm\) 5.28 (10.73)} &
\multicolumn{1}{c|}{\sircolor \textbf{10.85} \(\pm\) 4.27 (10.24)} &
\multicolumn{1}{c|}{\sircolor 5.36 \(\pm\) 5.25 (5.25)} &
\multicolumn{1}{c|}{\sircolor 5.66 \(\pm\) 3.49 (6.65)} &
\multicolumn{1}{c|}{\sircolor 7.96 \(\pm\) 5.22 (8.00)} & \multicolumn{1}{c|}{\sircolor SIR} \\ \cline{7-7}
\multicolumn{1}{|c|}{} &
\multicolumn{1}{c|} {\sarcolor 6.70 \(\pm\) 3.28 (7.21)} &
\multicolumn{1}{c|}{\sarcolor 7.31 \(\pm\) 3.23 (6.62)} &
\multicolumn{1}{c|}{\sarcolor \textbf{7.95} \(\pm\) 3.05 (7.93)} &
\multicolumn{1}{c|}{\sarcolor 5.11 \(\pm\) 2.65 (5.84)} &
\multicolumn{1}{c|}{\sarcolor 6.77 \(\pm\) 3.22 (6.71)} & \multicolumn{1}{c|}{\sarcolor SAR} \\ \hline \hline

\multicolumn{1}{|c|}{\multirow{3}{*}{\begin{tabular}[c]{@{}c@{}}DWA\end{tabular}}} & \multicolumn{1}{c|}{\sdrcolor 4.36 \(\pm\) 4.64 (5.24)} &
\multicolumn{1}{c|}{\sdrcolor \textbf{5.22} \(\pm\) 3.54 \textbf{(4.92)}} &
\multicolumn{1}{c|}{\sdrcolor \textbf{2.78} \(\pm\) 4.54 (2.88)} &
\multicolumn{1}{c|}{\sdrcolor 1.52 \(\pm\) 3.25 (2.45)} &
\multicolumn{1}{c|}{\sdrcolor 3.47 \(\pm\) 4.25 (3.61)} & \multicolumn{1}{c|}{\sdrcolor SDR} \\ \cline{7-7}
\multicolumn{1}{|c|}{} &
\multicolumn{1}{c|} {\sircolor 9.80 \(\pm\) 5.05 (10.65)} &
\multicolumn{1}{c|}{\sircolor 10.42 \(\pm\) 4.28 (10.11)} &
\multicolumn{1}{c|}{\sircolor 5.76 \(\pm\) 5.24 (5.63)} &
\multicolumn{1}{c|}{\sircolor 5.39 \(\pm\) 3.50 (6.48)} &
\multicolumn{1}{c|}{\sircolor 7.84 \(\pm\) 5.08 (7.78)} & \multicolumn{1}{c|}{\sircolor SIR} \\ \cline{7-7}
\multicolumn{1}{|c|}{} &
\multicolumn{1}{c|} {\sarcolor 6.78 \(\pm\) 3.35 \textbf{(7.30)}} &
\multicolumn{1}{c|}{\sarcolor \textbf{7.62} \(\pm\) 3.19 \textbf{(6.95)}} &
\multicolumn{1}{c|}{\sarcolor 7.81 \(\pm\) 3.04 (7.82)} &
\multicolumn{1}{c|}{\sarcolor \textbf{5.41} \(\pm\) 2.50 (\textbf{5.96})} &
\multicolumn{1}{c|}{\sarcolor \textbf{6.90} \(\pm\) 3.16 \textbf{(6.97)}} & \multicolumn{1}{c|}{\sarcolor SAR} \\ \hline \hline

\multicolumn{1}{|c|}{\multirow{3}{*}{\begin{tabular}[c]{@{}c@{}}EBW P1\end{tabular}}} &
\multicolumn{1}{c|}{\sdrcolor 4.51 \(\pm\) 4.56 (5.41)} &
\multicolumn{1}{c|}{\sdrcolor 5.13 \(\pm\) 3.50 (4.77)} &
\multicolumn{1}{c|}{\sdrcolor 2.64 \(\pm\) 4.32 (2.94)} &
\multicolumn{1}{c|}{\sdrcolor 1.59 \(\pm\) 3.17 \textbf{(2.64)}} &
\multicolumn{1}{c|}{\sdrcolor 3.46 \(\pm\) 4.15 (3.65)} & \multicolumn{1}{c|}{\sdrcolor SDR} \\ \cline{7-7}
\multicolumn{1}{|c|}{} &
\multicolumn{1}{c|} {\sircolor 10.25 \(\pm\) 4.97 (11.05)} &
\multicolumn{1}{c|}{\sircolor 10.60 \(\pm\) 4.34 (10.11)} &
\multicolumn{1}{c|}{\sircolor 5.38 \(\pm\) 4.91 (5.36)} &
\multicolumn{1}{c|}{\sircolor 5.64 \(\pm\) 3.31 (6.61)} &
\multicolumn{1}{c|}{\sircolor 7.97 \(\pm\) 5.04 (7.93)} & \multicolumn{1}{c|}{\sircolor SIR} \\ \cline{7-7}
\multicolumn{1}{|c|}{} &
\multicolumn{1}{c|} {\sarcolor 6.74 \(\pm\) 3.33 (7.29)} &
\multicolumn{1}{c|}{\sarcolor 7.37 \(\pm\) 3.18 (6.70)} &
\multicolumn{1}{c|}{\sarcolor 7.93 \(\pm\) 2.97 (7.99)} &
\multicolumn{1}{c|}{\sarcolor 5.27 \(\pm\) 2.63 (5.88)} &
\multicolumn{1}{c|}{\sarcolor 6.83 \(\pm\) 3.18 (6.77)} & \multicolumn{1}{c|}{\sarcolor SAR} \\ \hline \hline

\multicolumn{1}{|c|}{\multirow{3}{*}{\begin{tabular}[c]{@{}c@{}}EBW InstP1\end{tabular}}} &
\multicolumn{1}{c|}{\sdrcolor 4.49 \(\pm\) 4.62 (5.46)} &
\multicolumn{1}{c|}{\sdrcolor 5.16 \(\pm\) 3.55 (4.85)} &
\multicolumn{1}{c|}{\sdrcolor 2.63 \(\pm\) 4.53 (2.86)} &
\multicolumn{1}{c|}{\sdrcolor 1.58 \(\pm\) 3.18 (2.58)} &
\multicolumn{1}{c|}{\sdrcolor 3.46 \(\pm\) 4.24 (3.52)} & \multicolumn{1}{c|}{\sdrcolor SDR} \\ \cline{7-7}
\multicolumn{1}{|c|}{} &
\multicolumn{1}{c|} {\sircolor 10.31 \(\pm\) 5.10 (10.94)} &
\multicolumn{1}{c|}{\sircolor 10.61 \(\pm\) 4.24 (10.23)} &
\multicolumn{1}{c|}{\sircolor 5.52 \(\pm\) 5.39 (5.31)} &
\multicolumn{1}{c|}{\sircolor 5.61 \(\pm\) 3.41 (6.61)} &
\multicolumn{1}{c|}{\sircolor 8.01 \(\pm\) 5.08 (8.10)} & \multicolumn{1}{c|}{\sircolor SIR} \\ \cline{7-7}
\multicolumn{1}{|c|}{} &
\multicolumn{1}{c|} {\sarcolor 6.67 \(\pm\) 3.36 (7.19)} &
\multicolumn{1}{c|}{\sarcolor 7.39 \(\pm\) 3.28 (6.84)} &
\multicolumn{1}{c|}{\sarcolor 7.93 \(\pm\) 3.02 (7.90)} &
\multicolumn{1}{c|}{\sarcolor 5.28 \(\pm\) 2.57 (5.79)} &
\multicolumn{1}{c|}{\sarcolor 6.82 \(\pm\) 3.81 (6.86)} & \multicolumn{1}{c|}{\sarcolor SAR} \\ \hline \hline

\multicolumn{1}{|c|}{\multirow{3}{*}{\begin{tabular}[c]{@{}c@{}}Oh et al. \cite{oh2018spectrogram}\end{tabular}}} &
\multicolumn{1}{c|}{\sdrcolor 4.46 \(\pm\) 4.59 (5.29)} &
\multicolumn{1}{c|}{\sdrcolor 5.08 \(\pm\) 3.55 (4.86)} &
\multicolumn{1}{c|}{\sdrcolor 2.62 \(\pm\) 4.47 (2.85)} &
\multicolumn{1}{c|}{\sdrcolor \textbf{1.66} \(\pm\) 3.16 (2.55)} &
\multicolumn{1}{c|}{\sdrcolor 3.45 \(\pm\) 4.19 (3.60)} & \multicolumn{1}{c|}{\sdrcolor SDR} \\ \cline{7-7}
\multicolumn{1}{|c|}{} &
\multicolumn{1}{c|} {\sircolor 10.12 \(\pm\) 5.12 (10.88)} &
\multicolumn{1}{c|}{\sircolor 10.56 \(\pm\) 4.32 (10.25)} &
\multicolumn{1}{c|}{\sircolor 5.55 \(\pm\) 5.15 (5.15)} &
\multicolumn{1}{c|}{\sircolor 5.96 \(\pm\) 3.50 (6.75)} &
\multicolumn{1}{c|}{\sircolor 8.05 \(\pm\) 5.09 (8.24)} & \multicolumn{1}{c|}{\sircolor SIR} \\ \cline{7-7}
\multicolumn{1}{|c|}{} &
\multicolumn{1}{c|} {\sarcolor 6.77 \(\pm\) 3.20 (7.24)} &
\multicolumn{1}{c|}{\sarcolor 7.35 \(\pm\) 3.23 (6.72)} &
\multicolumn{1}{c|}{\sarcolor 7.77 \(\pm\) 3.13 (7.83)} &
\multicolumn{1}{c|}{\sarcolor 5.15 \(\pm\) 2.52 (5.74)} &
\multicolumn{1}{c|}{\sarcolor 6.76 \(\pm\) 3.17 (6.77)} & \multicolumn{1}{c|}{\sarcolor SAR} \\
\hline

\multicolumn{1}{|c|}{\multirow{3}{*}{\begin{tabular}[c]{@{}c@{}}EBW P2\end{tabular}}} &
\multicolumn{1}{c|}{\sdrcolor 4.48 \(\pm\) 4.82 (5.44)} &
\multicolumn{1}{c|}{\sdrcolor 5.17 \(\pm\) 3.60 (4.89)} &
\multicolumn{1}{c|}{\sdrcolor 2.69 \(\pm\) 4.44 (2.99)} &
\multicolumn{1}{c|}{\sdrcolor 1.62 \(\pm\) 3.14 (2.58)} &
\multicolumn{1}{c|}{\sdrcolor \textbf{3.49} \(\pm\) 4.26 \textbf{(3.66)}} & \multicolumn{1}{c|}{\sdrcolor SDR} \\ \cline{7-7}
\multicolumn{1}{|c|}{} &
\multicolumn{1}{c|} {\sircolor 10.20 \(\pm\) 5.27 (10.78)} &
\multicolumn{1}{c|}{\sircolor 10.40 \(\pm\) 4.03 (10.11)} &
\multicolumn{1}{c|}{\sircolor 5.74 \(\pm\) 5.45 (5.46)} &
\multicolumn{1}{c|}{\sircolor 5.72 \(\pm\) 3.39 \textbf{(6.77)}} &
\multicolumn{1}{c|}{\sircolor 8.01 \(\pm\) 5.12 (7.92)} & \multicolumn{1}{c|}{\sircolor SIR} \\ \cline{7-7}
\multicolumn{1}{|c|}{} &
\multicolumn{1}{c|} {\sarcolor 6.77 \(\pm\) 3.35 (7.28)} &
\multicolumn{1}{c|}{\sarcolor 7.48 \(\pm\) 3.30 (6.82)} &
\multicolumn{1}{c|}{\sarcolor 7.80 \(\pm\) 2.99 (7.93)} &
\multicolumn{1}{c|}{\sarcolor 5.26 \(\pm\) 2.50 (5.85)} &
\multicolumn{1}{c|}{\sarcolor 6.82 \(\pm\) 3.19 (6.79)} & \multicolumn{1}{c|}{\sarcolor SAR} \\

\hline

\end{tabular}%
}
\label{4srcresults}
\end{center}
\end{table*}
"""




"""
\begin{table}[tb]
\caption{Results of Singing Voice Separation}
\resizebox{0.99\linewidth}{!}{%
\LARGE
\begin{tabular}{|ccccc}
\cline{1-4}
\multicolumn{1}{|c|}{Model} & \multicolumn{1}{c|}{Vocals} & \multicolumn{1}{c|}{Accompaniment} & \multicolumn{1}{c|}{Overall} &  \\ \hline
\multicolumn{1}{|c|}{\multirow{3}{*}{\begin{tabular}[c]{@{}c@{}}Dedicated  \\ U-Nets (x2)\end{tabular}}} & \multicolumn{1}{c|}{\sdrcolor 5.09 \(\pm\) 4.31 (5.61)} & \multicolumn{1}{c|}{\sdrcolor 12.95 \(\pm\) 3.18 (12.53)} & \multicolumn{1}{c|}{\sdrcolor 9.02 \(\pm\) 5.46 (9.64)} & \multicolumn{1}{c|}{\sdrcolor SDR} \\ \cline{5-5} 
\multicolumn{1}{|c|}{} & \multicolumn{1}{c|}{\sircolor 11.68 \(\pm\) 4.85 (12.02)} & \multicolumn{1}{c|}{\sircolor 17.80 \(\pm\) 3.80 (17.60)} & \multicolumn{1}{c|}{\sircolor 14.74 \(\pm\) 5.31 (14.93)} & \multicolumn{1}{c|}{\sircolor SIR} \\ \cline{5-5} 
\multicolumn{1}{|c|}{} & \multicolumn{1}{c|}{\sarcolor 6.83 \(\pm\) 3.36 (7.20)} & \multicolumn{1}{c|}{\sarcolor 15.07 \(\pm\) 3.35 (14.82)} & \multicolumn{1}{c|}{\sarcolor 10.95 \(\pm\) 5.32 (11.16)} & \multicolumn{1}{c|}{\sarcolor SAR} \\ \hline \hline \hline
 
\multicolumn{1}{|c|}{\multirow{3}{*}{\begin{tabular}[c]{@{}c@{}}C-U-Net\end{tabular}}} & \multicolumn{1}{c|}{\sdrcolor 4.42 \(\pm\) 4.98 (5.17)} & \multicolumn{1}{c|}{\sdrcolor 12.21 \(\pm\) 2.58 (12.16)} & \multicolumn{1}{c|}{\sdrcolor 8.31 \(\pm\) 5.56 (9.26)} & \multicolumn{1}{c|}{\sdrcolor SDR} \\ \cline{5-5} 
\multicolumn{1}{|c|}{} & \multicolumn{1}{c|}{\sircolor \textbf{12.99} \(\pm\) 5.67 \textbf{(13.93)}} & \multicolumn{1}{c|}{\sircolor 18.16 \(\pm\) 4.13 (17.65)} & \multicolumn{1}{c|}{\sircolor \textbf{15.57} \(\pm\) 5.58 \textbf{(16.12)}} & \multicolumn{1}{c|}{\sircolor SIR} \\ \cline{5-5} 
\multicolumn{1}{|c|}{} & \multicolumn{1}{c|}{\sarcolor 5.73 \(\pm\) 3.64 (5.92)} & \multicolumn{1}{c|}{\sarcolor 13.94 \(\pm\) 2.69 (14.06)} & \multicolumn{1}{c|}{\sarcolor 9.83 \(\pm\) 5.21 (9.99)} & \multicolumn{1}{c|}{\sarcolor SAR} \\ \hline \hline

\multicolumn{1}{|c|}{\multirow{3}{*}{\begin{tabular}[c]{@{}c@{}}UW\end{tabular}}} & \multicolumn{1}{c|}{\sdrcolor 5.06 \(\pm\) 4.93 (5.75)} & \multicolumn{1}{c|}{\sdrcolor 12.98 \(\pm\) 3.14 (12.48)} & \multicolumn{1}{c|}{\sdrcolor 9.02 \(\pm\) 5.72 (9.74)} & \multicolumn{1}{c|}{\sdrcolor SDR} \\ \cline{5-5} 
\multicolumn{1}{|c|}{} & \multicolumn{1}{c|}{\sircolor 11.84 \(\pm\) 5.09 (12.00)} & \multicolumn{1}{c|}{\sircolor 17.60 \(\pm\) 3.81 (17.24)} & \multicolumn{1}{c|}{\sircolor 14.72 \(\pm\) 5.33 (15.00)} & \multicolumn{1}{c|}{\sircolor SIR} \\ \cline{5-5} 
\multicolumn{1}{|c|}{} & \multicolumn{1}{c|}{\sarcolor 6.81 \(\pm\) 4.02 (7.31)} & \multicolumn{1}{c|}{\sarcolor 15.20 \(\pm\) 3.18 (15.01)} & \multicolumn{1}{c|}{\sarcolor 11.00 \(\pm\) 5.55 (11.24)} & \multicolumn{1}{c|}{\sarcolor SAR} \\ \hline \hline

\multicolumn{1}{|c|}{\multirow{3}{*}{\begin{tabular}[c]{@{}c@{}}DWA\end{tabular}}} & \multicolumn{1}{c|}{\sdrcolor 5.20 \(\pm\) 4.50 (5.67)} & \multicolumn{1}{c|}{\sdrcolor 12.96 \(\pm\) 3.11 (12.44)} & \multicolumn{1}{c|}{\sdrcolor 9.08 \(\pm\) 5.48 (9.61)} & \multicolumn{1}{c|}{\sdrcolor SDR} \\ \cline{5-5} 
\multicolumn{1}{|c|}{} & \multicolumn{1}{c|}{\sircolor 11.97 \(\pm\) 4.81 (12.30)} & \multicolumn{1}{c|}{\sircolor 17.15 \(\pm\) 3.83 (16.67)} & \multicolumn{1}{c|}{\sircolor 14.56 \(\pm\) 5.05 (14.67)} & \multicolumn{1}{c|}{\sircolor SIR} \\ \cline{5-5} 
\multicolumn{1}{|c|}{} & \multicolumn{1}{c|}{\sarcolor 6.86 \(\pm\) 3.66 (7.44)} & \multicolumn{1}{c|}{\sarcolor 15.44 \(\pm\) 3.03 \textbf{(15.21)}} & \multicolumn{1}{c|}{\sarcolor 11.15 \(\pm\) 5.46 (11.54)} & \multicolumn{1}{c|}{\sarcolor SAR} \\ \hline \hline

\multicolumn{1}{|c|}{\multirow{3}{*}{\begin{tabular}[c]{@{}c@{}}EBW P1\end{tabular}}} & \multicolumn{1}{c|}{\sdrcolor 5.12 \(\pm\) 4.78 \textbf{(5.89)}} & \multicolumn{1}{c|}{\sdrcolor \textbf{13.06} \(\pm\) 2.91 \textbf{(12.88)}} & \multicolumn{1}{c|}{\sdrcolor 9.09 \(\pm\) 5.60 (9.77)} & \multicolumn{1}{c|}{\sdrcolor SDR} \\ \cline{5-5} 
\multicolumn{1}{|c|}{} & \multicolumn{1}{c|}{\sircolor 11.81 \(\pm\) 5.15 (12.08)} & \multicolumn{1}{c|}{\sircolor \textbf{18.23} \(\pm\) 3.88 \textbf{(17.84)}} & \multicolumn{1}{c|}{\sircolor 15.02 \(\pm\) 5.57 (15.41)} & \multicolumn{1}{c|}{\sircolor SIR} \\ \cline{5-5} 
\multicolumn{1}{|c|}{} & \multicolumn{1}{c|}{\sarcolor 6.86 \(\pm\) 3.62 (7.36)} & \multicolumn{1}{c|}{\sarcolor 14.99 \(\pm\) 2.85 (15.00)} & \multicolumn{1}{c|}{\sarcolor 10.92 \(\pm\) 5.21 (11.02)} & \multicolumn{1}{c|}{\sarcolor SAR} \\ \hline \hline

\multicolumn{1}{|c|}{\multirow{3}{*}{\begin{tabular}[c]{@{}c@{}}EBW InstP1\end{tabular}}} & \multicolumn{1}{c|}{\sdrcolor \textbf{5.28} \(\pm\) 4.60 (5.79)} & \multicolumn{1}{c|}{\sdrcolor 13.04 \(\pm\) 3.02 (12.69)} & \multicolumn{1}{c|}{\sdrcolor \textbf{9.16} \(\pm\) 5.50 \textbf{(9.79)}} & \multicolumn{1}{c|}{\sdrcolor SDR} \\ \cline{5-5} 
\multicolumn{1}{|c|}{} & \multicolumn{1}{c|}{\sircolor 12.01 \(\pm\) 5.24 (12.22)} & \multicolumn{1}{c|}{\sircolor 17.36 \(\pm\) 3.91 (16.85)} & \multicolumn{1}{c|}{\sircolor 14.69 \(\pm\) 5.33 (14.93)} & \multicolumn{1}{c|}{\sircolor SIR} \\ \cline{5-5} 
\multicolumn{1}{|c|}{} & \multicolumn{1}{c|}{\sarcolor \textbf{7.02} \(\pm\) 3.41 \textbf{(7.46)}} & \multicolumn{1}{c|}{\sarcolor \textbf{15.45} \(\pm\) 2.88 (15.20)} & \multicolumn{1}{c|}{\sarcolor \textbf{11.24} \(\pm\) 5.27 \textbf{(11.59)}} & \multicolumn{1}{c|}{\sarcolor SAR} \\ \hline \hline

\multicolumn{1}{|c|}{\multirow{3}{*}{\begin{tabular}[c]{@{}c@{}}Oh et al. \cite{oh2018spectrogram}\end{tabular}}} & \multicolumn{1}{c|}{\sdrcolor 5.18 \(\pm\) 4.17 (5.67)} & \multicolumn{1}{c|}{\sdrcolor 13.00 \(\pm\) 3.03 (12.63)} & \multicolumn{1}{c|}{\sdrcolor 9.09 \(\pm\) 5.35 (9.78)} & \multicolumn{1}{c|}{\sdrcolor SDR} \\ \cline{5-5} 
\multicolumn{1}{|c|}{} & \multicolumn{1}{c|}{\sircolor 11.30 \(\pm\) 5.12 (12.12)} & \multicolumn{1}{c|}{\sircolor 17.31 \(\pm\) 3.95 (17.60)} & \multicolumn{1}{c|}{\sircolor 15.00 \(\pm\) 5.27 (15.30)} & \multicolumn{1}{c|}{\sircolor SIR} \\ \cline{5-5} 
\multicolumn{1}{|c|}{} & \multicolumn{1}{c|}{\sarcolor 7.00 \(\pm\) 3.41 (7.19)} & \multicolumn{1}{c|}{\sarcolor 15.27 \(\pm\) 2.84 (14.84)} & \multicolumn{1}{c|}{\sarcolor 10.91 \(\pm\) 5.20 (11.04)} & \multicolumn{1}{c|}{\sarcolor SAR} \\ \hline

\multicolumn{1}{|c|}{\multirow{3}{*}{\begin{tabular}[c]{@{}c@{}}EBW P2$^{*}$\end{tabular}}} & \multicolumn{1}{c|}{\sdrcolor 5.07 \(\pm\) 4.56 (5.63)} & \multicolumn{1}{c|}{\sdrcolor 12.89 \(\pm\) 2.95 (12.39)} & \multicolumn{1}{c|}{\sdrcolor 8.98 \(\pm\) 5.48 (9.66)} & \multicolumn{1}{c|}{\sdrcolor SDR} \\ \cline{5-5} 
\multicolumn{1}{|c|}{} & \multicolumn{1}{c|}{\sircolor 11.30 \(\pm\) 5.12 (11.47)} & \multicolumn{1}{c|}{\sircolor 17.31 \(\pm\) 3.95 (16.92)} & \multicolumn{1}{c|}{\sircolor 14.30 \(\pm\) 5.46 (14.56)} & \multicolumn{1}{c|}{\sircolor SIR} \\ \cline{5-5} 
\multicolumn{1}{|c|}{} & \multicolumn{1}{c|}{\sarcolor 7.00 \(\pm\) 3.41 (7.31)} & \multicolumn{1}{c|}{\sarcolor 15.27 \(\pm\) 2.84 (15.04)} & \multicolumn{1}{c|}{\sarcolor 11.13 \(\pm\) 5.20 (11.42)} & \multicolumn{1}{c|}{\sarcolor SAR} \\ \hline

\multicolumn{4}{l}{$^{*}$ trained with learning rate 0.001 instead of 0.01.}
\end{tabular}%
}
\label{2srcresults}
\end{table}
"""


"""
#### 4src results only SDR
\begin{table*}[tb]
\begin{center}
\caption{Results of Multi-instrument Source Separation in SDR (median in parenthesis)}
\resizebox{0.7\textwidth}{!}{%
\LARGE
\begin{tabular}{|c|c|c|c|c|c|}
\cline{1-6}
Model & Vocals & Drums & Bass & Rest & Overall  \\ \hline
\begin{tabular}[c]{@{}c@{}}Dedicated  \\ U-Nets (x4)\end{tabular} & 
\textbf{4.96} \(\pm\) 4.63 \textbf{(5.77)} & 
4.95 \(\pm\) 3.56 (4.60) & 
\textbf{2.78} \(\pm\) 4.41 \textbf{(3.19)} & 
1.21 \(\pm\) 3.38 (2.23) & 
3.48 \(\pm\) 4.30 (3.61)
\\ \hline \hline \hline

C-U-Net & 4.49 \(\pm\) 4.75 (5.26) & 4.54 \(\pm\) 3.59 (4.30) & 
2.51 \(\pm\) 4.26 (2.97) & 0.97 \(\pm\) 3.57 (1.69) & 
3.13 \(\pm\) 4.31 (3.37) 
\\ \hline \hline

UW & 4.31 \(\pm\) 4.80 (5.46) & 5.19 \(\pm\) 3.51 (4.72) & 
2.55 \(\pm\) 4.58 (2.81) &  1.51 \(\pm\) 3.32 (2.49) & 
3.39 \(\pm\) 4.32 (3.58)
\\ \hline \hline

DWA & 4.36 \(\pm\) 4.64 (5.24) & \textbf{5.22} \(\pm\) 3.54 \textbf{(4.92)} & 
\textbf{2.78} \(\pm\) 4.54 (2.88) & 1.52 \(\pm\) 3.25 (2.45) & 
3.47 \(\pm\) 4.25 (3.61)  
\\ \hline \hline 

EBW P1  & 4.51 \(\pm\) 4.56 (5.41) & 5.13 \(\pm\) 3.50 (4.77) & 
2.64 \(\pm\) 4.32 (2.94) &  1.59 \(\pm\) 3.17 \textbf{(2.64)} & 
3.46 \(\pm\) 4.15 (3.65) 
\\ \hline \hline 

EBW InstP1 & 
4.49 \(\pm\) 4.62 (5.46) & 
5.16 \(\pm\) 3.55 (4.85) & 
2.63 \(\pm\) 4.53 (2.86) & 
1.58 \(\pm\) 3.18 (2.58) & 
3.46 \(\pm\) 4.24 (3.52)
\\ \hline \hline 

Oh et al. \cite{oh2018spectrogram} & 
4.46 \(\pm\) 4.59 (5.29) & 
5.08 \(\pm\) 3.55 (4.86) & 
2.62 \(\pm\) 4.47 (2.85) & 
\textbf{1.66} \(\pm\) 3.16 (2.55) & 
3.45 \(\pm\) 4.19 (3.60)
\\  \hline \hline

EBW P2 & 4.48 \(\pm\) 4.82 (5.44) & 5.17 \(\pm\) 3.60 (4.89) & 
2.69 \(\pm\) 4.44 (2.99) & 1.62 \(\pm\) 3.14 (2.58) & 
\textbf{3.49} \(\pm\) 4.26 \textbf{(3.66)}
\\ \hline \hline

\end{tabular}%
}
\label{4srcresults}
\end{center}
\end{table*}

"""