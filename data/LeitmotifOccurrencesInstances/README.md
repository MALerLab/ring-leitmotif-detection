Dataset for the paper "Towards Leitmotif Activity Detection in Opera Recordings" by Krause et al., submitted
Please refer to the paper website https://www.audiolabs-erlangen.de/resources/MIR/2021-TISMIR-TowardsLeitmotifDetection for details.

This dataset constitutes a strict superset of the data found on https://www.audiolabs-erlangen.de/resources/MIR/2020-ISMIR-LeitmotifClassification.

In this dataset, we use the following naming scheme to refer to different parts of the Ring:

Name | Description
--- | ---
A | Das Rheingold
B-1 | Die Walküre, Act 1
B-2 | Die Walküre, Act 2
B-3 | Die Walküre, Act 2
C-1 | Siegfried, Act 1
C-2 | Siegfried, Act 2
C-3 | Siegfried, Act 3
D-0 | Götterdämmerung, Vorspiel
D-1 | Götterdämmerung, Act 1
D-2 | Götterdämmerung, Act 2
D-3 | Götterdämmerung, Act 3

#### Occurrences

Occurrence positions are found in the .csv-files in the "Occurrences" subfolder of the zip-archive. For example, "Occurrences/B-2.csv" contains all motif occurrences in Die Walküre, Act 2. In these files, each line corresponds to a motif occurrence. Start and end positions are given in measures. For example, the line

```
Ring;778.5;780.25
```

in "Occurrences/B-2.csv" signifies that there is an occurrence of the Ring motif starting at 778.5 (a half measure after 778) and ending at 780.25 (a quarter measure after measure 780). Measure numbers correspond to the piano score from Richard Kleinmichel, available at (link:https://www.imslp.org text: IMSLP).  

#### Instances

Instance positions are found in the .csv-files inside the 16 directories of the "Instance" subfolder of the zip-archive. For example, "Instances/Wagner_RingBarenboimKupfer_WC2009/B-2.csv" contains all motif instances in the Daniel Barenboim performance of Die Walküre, Act 2. In these files, each line corresponds to a motif instance. Start and end positions are given in seconds (for this, all CD tracks for a particular performance of an act have been cut and concatenated to form one continuous audio file for that act). For example, the line

```
Ring;2130.6;2137.2
```

in "Instances/Wagner_RingBarenboimKupfer_WC2009/B-2.csv" signifies that there is an instance of the Ring motif starting at second 2130.6 and ending at 2137.2.

A table of the performances used in this study, including ID, conductors, years of recording and length, can be found in the paper. The following table provides helpful information for identifying the exact CD releases:

?? Different release
== Found
// Included
!! Can't find

ID in Paper | Label | Year of release | Conductor, Orchestra, Choir
--- | --- | --- | ---
??P-Ba | WC | 2009 | Daniel Barenboim, Chor und Orchester der Bayreuther Festspiele
??P-Ha | EMI | 2008 | Bernard Haitink, Symphonieorchester und Chor des Bayrischen Rundfunks
==P-Ka | DG | 1998 | Herbert von Karajan, Berliner Philharmoniker, Chor der Deutschen Oper Berlin
!!P-Sa | EMI | 2012 | Wolfgang Sawallisch, Bayrisches Staatsorchester, Chor der Bayrischen Staatsoper
??P-So | DECCA | 2012 | Georg Solti, Wiener Staatsopernchor, Wiener Philharmoniker
==P-We | OEHMS | 2013 | Sebastian Weigle, Frankfurter Opern- und Museumsorchester, Chor und Herren des Extrachores der Oper Frankfurt
==P-Bo | PHILIPS | 2006 | Pierre Boulez, Chor und Orchester der Bayreuther Festspiele
??P-Bö | DECCA | 2008 | Karl Böhm, Chor und Orchester der Bayreuther Festspiele
//P-Fu | EMI | 2011 | Wilhelm Furtwängler, Orchestra Sinfonica della Radio Italiana, Coro della Radio Italiana
??P-Ja | SONY | 2012 | Marek Janowski, Staatskapelle Dresden, Männer des Staatsopernchores Leipzig, Staatsopernchor Dresden
//P-Ke | ZYX | 2012 | Joseph Keilberth, Chor und Orchester der Bayreuther Festspiele; Wilhelm Furtwängler, Wiener Philharmoniker
//P-Kr | ORFEO | 2010 | Clemens Krauss, Chor und Orchester der Bayreuther Festspiele
??P-Le | DG | 2012 | James Levine, The Metropolitan Opera Orchestra, The Metropolitan Opera Chorus
!!P-Ne | MEMBRAN | 1995 | Günther Neuhold, Badische Staatskapelle, Badischer Staatsopernchor
==P-Sw | PROFIL | 2013 | Hans Swarowsky, Grosses Symphonieorchester mit Mitgliedern der Tschechischen Philharmonie und des Orchesters des Nationaltheaters Prag, Chor der Wiener Volksoper
==P-Th | DG | 2013 | Christian Thielemann, Chor und Zusatzchor der Wiener Staatsoper, Orchester der Wiener Staatsoper, Bühnenorchester der Wiener Staatsoper

Thus, the performance by Wolfgang Sawallisch (P-Sa) was released by EMI in 2012.
