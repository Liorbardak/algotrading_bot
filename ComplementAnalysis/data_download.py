import yfinance as yf
import pandas as pd
import os
import numpy as np
import pylab as plt
stockdir =  'C:/Users/dadab/projects/algotrading/data/tickers'
outdir =  'C:/Users/dadab/projects/algotrading/data/snp500_yahoo'
bad = []
bad_length = []
snp = pd.read_csv(os.path.join(stockdir, 'sp500_stocks.csv'))

start = '2019-01-02'
end = '2025-06-01'

# datadir = 'C:/Users/dadab/projects/algotrading/data/stocks_quality/analysis/analysis'
# compliments_stocks = [name for name in os.listdir(datadir) if os.path.isfile(os.path.join(datadir, name ,'summarizeCompliments.json'))]

#stocks_to_run_on = set(snp.Ticker.values).union(set(compliments_stocks))
#stocks_to_run_on = ['ADN']
#stocks_to_run_on = sorted([entry for entry in os.listdir(stockdir) if os.path.isdir( os.path.join(stockdir, entry))])
bad_stocks = ['AACG', 'AAON', 'ABEO', 'ABL', 'ABLV', 'ABTS', 'ABUS', 'ABVC', 'ABVE', 'AC', 'ACAD', 'ACB', 'ACEL', 'ACET', 'ACHV', 'ACIU', 'ACLO', 'ACON', 'ACR', 'ACRS', 'ACRV', 'ADCT', 'ADD', 'ADEA', 'ADGM', 'ADIL', 'ADN', 'ADSE', 'ADTX', 'ADV', 'ADVM', 'ADXN', 'AEAEU', 'AEHL', 'AEI', 'AEMD', 'AENT', 'AEON', 'AEVA', 'AFBI', 'AFCG', 'AFGB', 'AFGC', 'AFGD', 'AFGE', 'AFJK', 'AFMD', 'AFRI', 'AFRM', 'AFYA', 'AGAE', 'AGEN', 'AGFY', 'AGM-A', 'AGMH', 'AGNCL', 'AGNCM', 'AGS', 'AHCO', 'AHG', 'AHI', 'AHT', 'AIFF', 'AIFU', 'AIHS', 'AILE', 'AIM', 'AIMD', 'AINP', 'AIRI', 'AIRT', 'AIRTP', 'AITRU', 'AIV', 'AIXI', 'AKA', 'AKAN', 'AKBA', 'AKO-A', 'AKRO', 'AKTX', 'ALAR', 'ALBT', 'ALCE', 'ALCY', 'ALCYU', 'ALDX', 'ALGS', 'ALLK', 'ALLO', 'ALLR', 'ALNT', 'ALRS', 'ALTI', 'ALTS', 'ALX', 'ALXO', 'ALZN', 'AM', 'AMBO', 'AMC', 'AMCR', 'AMIX', 'AMJB', 'AMLX', 'AMOD', 'AMPG', 'AMPGW', 'AMPY', 'AMR', 'AMRN', 'AMST', 'AMTB', 'AMTD', 'AMWL', 'AMZN', 'ANAB', 'ANGH', 'ANGI', 'ANRO', 'ANTE', 'ANTX', 'ANY', 'AONC', 'APCX', 'APDN', 'APEI', 'APLD', 'APLM', 'APLT', 'APM', 'APPS', 'APRE', 'APVO', 'APYX', 'AQB', 'AQMS', 'ARBB', 'ARBE', 'ARBK', 'ARBKL', 'ARCT', 'ARDX', 'AREB', 'AREC', 'AREN', 'ARKO', 'ARLO', 'ARMK', 'ARMN', 'ARMP', 'AROW', 'ARQQ', 'ARQQW', 'ARR', 'ARRY', 'ARTL', 'ASMB', 'ASNS', 'ASPN', 'ASPS', 'ASR', 'ASRT', 'ASST', 'ASTC', 'ASTI', 'ASTL', 'ASTS', 'ATCH', 'ATCOL', 'ATHA', 'ATHE', 'ATLN', 'ATNF', 'ATNM', 'ATPC', 'ATRA', 'ATS', 'ATXG', 'ATXI', 'ATXS', 'ATYR', 'AUID', 'AUUD', 'AVGR', 'AVIR', 'AVO', 'AVTE', 'AVTX', 'AXDX', 'AXSM', 'AYI', 'AYRO', 'AYTU', 'AZ', 'AZTR', 'B', 'BACK', 'BAFN', 'BAK', 'BALY', 'BANL', 'BANX', 'BAOS', 'BATL', 'BBAI', 'BBAR', 'BBD', 'BBGI', 'BBIO', 'BBLG', 'BBLGW', 'BBU', 'BBWI', 'BCAB', 'BCAL', 'BCDA', 'BCLI', 'BCOW', 'BCRX', 'BCSF', 'BCTX', 'BDCX', 'BDRX', 'BDTX', 'BDX', 'BE', 'BECN', 'BEDU', 'BEEM', 'BEEP', 'BENF', 'BEP', 'BERY', 'BERZ', 'BFH', 'BFRI', 'BGI', 'BGLC', 'BH-A', 'BHAT', 'BHC', 'BHFAM', 'BHFAO', 'BHP', 'BHR', 'BHRB', 'BIO-B', 'BIOA', 'BIOX', 'BIP', 'BIRD', 'BITF', 'BIVI', 'BJDX', 'BKDV', 'BKHAU', 'BKKT', 'BKR', 'BKSY', 'BKTI', 'BKYI', 'BLBX', 'BLIN', 'BLRX', 'BLUE', 'BMA', 'BMRA', 'BN', 'BNED', 'BNGO', 'BNH', 'BNL', 'BNR', 'BNRG', 'BNTC', 'BNZI', 'BOF', 'BOLT', 'BON', 'BORR', 'BOTJ', 'BOWN', 'BOXL', 'BPMC', 'BPOPM', 'BPTH', 'BPYPO', 'BQ', 'BRAG', 'BRBR', 'BRBS', 'BRCC', 'BREA', 'BRFH', 'BRID', 'BRIF', 'BRK-A', 'BRKD', 'BRKHU', 'BROG', 'BRTX', 'BRW', 'BSLK', 'BTAI', 'BTBD', 'BTBT', 'BTC', 'BTCM', 'BTCS', 'BTCT', 'BTDR', 'BTE', 'BTM', 'BTMD', 'BTOG', 'BTTR', 'BULZ', 'BUR', 'BVFL', 'BW', 'BWA', 'BWIN', 'BWLP', 'BWMX', 'BXC', 'BYFC', 'BYRN', 'BYSI', 'BZFD', 'CALC', 'CANB', 'CANF', 'CANG', 'CAPR', 'CARD', 'CARM', 'CARU', 'CASI', 'CATX', 'CAUD', 'CAVA', 'CBSH', 'CCCS', 'CCEL', 'CCG', 'CCGWW', 'CCI', 'CCLDO', 'CCM', 'CCSI', 'CCTSU', 'CCZ', 'CDIO', 'CDLR', 'CDRO', 'CDTG', 'CDTX', 'CDXS', 'CEAD', 'CEFD', 'CEG', 'CELC', 'CELU', 'CELZ', 'CENT', 'CENTA', 'CEPU', 'CETX', 'CETY', 'CFLT', 'CGBDL', 'CGC', 'CGNT', 'CGTX', 'CHE', 'CHEB', 'CHEK', 'CHGG', 'CHNR', 'CHRD', 'CHTR', 'CIFR', 'CIG', 'CIG-C', 'CIM', 'CING', 'CISO', 'CISS', 'CJET', 'CKPT', 'CKX', 'CLBT', 'CLCO', 'CLDI', 'CLDX', 'CLEU', 'CLNN', 'CLPT', 'CLRB', 'CLRO', 'CLSK', 'CLYM', 'CMCM', 'CMCT', 'CMG', 'CMMB', 'CMND', 'CMPO', 'CMT', 'CNDT', 'CNET', 'CNEY', 'CNFRZ', 'CNOBP', 'CNQ', 'CNSP', 'CNVS', 'CNXC', 'COCP', 'COE', 'COEP', 'COFS', 'COGT', 'COKE', 'COSM', 'CPAC', 'CPHI', 'CPOP', 'CPRI', 'CRBG', 'CRBP', 'CRCT', 'CRDF', 'CRDL', 'CRDO', 'CREG', 'CRESY', 'CREX', 'CRGX', 'CRKN', 'CRMD', 'CRVO', 'CSCI', 'CSLR', 'CSR', 'CSWI', 'CTGO', 'CTHR', 'CTLP', 'CTM', 'CTMX', 'CTNT', 'CTO', 'CTRI', 'CTRM', 'CTXR', 'CUBT', 'CURI', 'CUZ', 'CVEO', 'CVKD', 'CVM', 'CVR', 'CWD', 'CXDO', 'CXT', 'CYBN', 'CYCC', 'CYCCP', 'CYCN', 'CYN', 'CYTK', 'CZFS', 'DAC', 'DADA', 'DALN', 'DAO', 'DARE', 'DATS', 'DATSW', 'DAVA', 'DAVE', 'DAWN', 'DBGI', 'DBRG', 'DBVT', 'DCBO', 'DCGO', 'DCTH', 'DD', 'DDC', 'DEA', 'DEC', 'DECK', 'DELL', 'DERM', 'DESP', 'DEVS', 'DFLI', 'DGICB', 'DGLY', 'DHAI', 'DHR', 'DIT', 'DKNG', 'DLNG', 'DLO', 'DLPN', 'DNA', 'DOC', 'DOCU', 'DOGZ', 'DOMH', 'DOUG', 'DOYU', 'DPRO', 'DRMA', 'DRRX', 'DRUG', 'DRVN', 'DSGN', 'DSS', 'DTCK', 'DTE', 'DTG', 'DTIL', 'DTM', 'DTSS', 'DTST', 'DUOL', 'DVQQ', 'DWTX', 'DX', 'DYN', 'EBON', 'ECCC', 'ECCX', 'ECO', 'ECOR', 'ECVT', 'ECX', 'EDBL', 'EDSA', 'EDU', 'EFOI', 'EFTR', 'EFXT', 'EH', 'EHAB', 'EHC', 'EICA', 'EICB', 'EJH', 'EKSO', 'ELAB', 'ELBM', 'ELC', 'ELDN', 'ELEV', 'ELLO', 'ELP', 'ELTX', 'ELUT', 'ELVA', 'ELVN', 'ELWS', 'EMBC', 'EMP', 'ENFY', 'ENG', 'ENLT', 'ENLV', 'ENOV', 'ENSC', 'ENSG', 'ENTO', 'ENVB', 'ENVX', 'EOSE', 'EOSEW', 'EP', 'EPAM', 'EPIX', 'EPM', 'EPSN', 'EQX', 'ERNA', 'ERO', 'ESE', 'ESEA', 'ESGR', 'ESHA', 'ESLA', 'ESOA', 'ESPR', 'ETH', 'ETI-P', 'EU', 'EUDA', 'EVAX', 'EVC', 'EVEX', 'EVGN', 'EVGOW', 'EVGR', 'EVH', 'EVLV', 'EVO', 'EVOK', 'EVTL', 'EVTL-WS', 'EVTV', 'EXC', 'EXEEL', 'EXEEW', 'EXEEZ', 'EYEN', 'EYPT', 'EZGO', 'FAMI', 'FARO', 'FAST', 'FAT', 'FATBB', 'FATBP', 'FBIN', 'FBIO', 'FBIOP', 'FBLG', 'FBRX', 'FBYD', 'FCEL', 'FCNCO', 'FCNCP', 'FCRX', 'FCUV', 'FEAM', 'FEDU', 'FENC', 'FENG', 'FERG', 'FET', 'FGBI', 'FGBIP', 'FGEN', 'FGF', 'FGFPP', 'FIP', 'FISK', 'FKWL', 'FLEX', 'FLG', 'FLGC', 'FLIC', 'FLNT', 'FLUX', 'FLYD', 'FLYE', 'FLYX', 'FMC', 'FMST', 'FNF', 'FNGR', 'FNGS', 'FNWD', 'FOA', 'FORD', 'FORTY', 'FOXO', 'FRAF', 'FRGE', 'FRGT', 'FRLAU', 'FRMEP', 'FRSX', 'FRZA', 'FSEA', 'FSK', 'FTAI', 'FTAIM', 'FTAIN', 'FTCI', 'FTEL', 'FTFT', 'FTI', 'FTK', 'FTLF', 'FTRE', 'FTV', 'FUBO', 'FULC', 'FURY', 'FVCB', 'FWONA', 'FWONK', 'FXNC', 'GAME', 'GANX', 'GB', 'GBDC', 'GBIO', 'GBTG', 'GCMG', 'GCTK', 'GDC', 'GDEV', 'GDHG', 'GDOT', 'GDSTU', 'GDXD', 'GDYN', 'GE', 'GECC', 'GECCI', 'GEG', 'GEHC', 'GENI', 'GEV', 'GFAI', 'GGAL', 'GGB', 'GHC', 'GHI', 'GHM', 'GIFT', 'GIII', 'GJO', 'GJP', 'GJR', 'GJS', 'GJT', 'GL', 'GLACU', 'GLAD', 'GLBS', 'GLMD', 'GLSTU', 'GLTO', 'GLXG', 'GLYC', 'GMFI', 'GMFIU', 'GMGI', 'GMGT', 'GNFT', 'GNLN', 'GNLX', 'GNPX', 'GNS', 'GOCO', 'GOEV', 'GOLLQ', 'GOOG', 'GOOGL', 'GOSS', 'GOTU', 'GOVX', 'GP', 'GPCR', 'GPN', 'GPOR', 'GRABW', 'GRAF-U', 'GRAL', 'GREE', 'GRFX', 'GRI', 'GRMN', 'GRNQ', 'GRNT', 'GROV', 'GRPN', 'GRRR', 'GRVY', 'GRWG', 'GRYP', 'GSAT', 'GSK', 'GSL', 'GSUN', 'GTBP', 'GTE', 'GTEC', 'GTX', 'GURE', 'GV', 'GWAV', 'GWH', 'GXAI', 'GYRE', 'HAFN', 'HBANL', 'HBANP', 'HBCP', 'HCTI', 'HCWB', 'HEPA', 'HFBL', 'HFFG', 'HG', 'HGBL', 'HGLB', 'HHH', 'HHS', 'HIPO', 'HITI', 'HIVE', 'HKD', 'HLLY', 'HLN', 'HLVX', 'HNVR', 'HODL', 'HOFV', 'HOLO', 'HON', 'HONE', 'HOOK', 'HOTH', 'HOV', 'HOVR', 'HPH', 'HPK', 'HPKEW', 'HQI', 'HR', 'HROW', 'HROWM', 'HSCS', 'HSDT', 'HSHP', 'HSIC', 'HSON', 'HSPOU', 'HTCO', 'HTFB', 'HTFC', 'HTOOW', 'HTZ', 'HUBC', 'HUDAR', 'HUDI', 'HUIZ', 'HUMA', 'HUN', 'HURA', 'HUSA', 'HUT', 'HVT-A', 'HWBK', 'HYFM', 'HYLN', 'HYMC', 'HYMCL', 'HYPR', 'HZO', 'IAC', 'IAS', 'IAUX', 'IBIO', 'IBKR', 'IBM', 'IBTA', 'ICCM', 'ICCT', 'ICG', 'ICON', 'ICU', 'IDA', 'IDAI', 'IDN', 'IDR', 'IFBD', 'IFED', 'IFS', 'IGIC', 'IGMS', 'IGTAU', 'IHG', 'IHRT', 'IINNW', 'IKNA', 'IKT', 'ILLR', 'ILMN', 'IMCC', 'IMG', 'IMMP', 'IMNN', 'IMPP', 'IMTE', 'IMTX', 'IMUX', 'IMVT', 'INAB', 'INBS', 'INBX', 'INCR', 'INDI', 'INDO', 'INDP', 'INDV', 'INHD', 'INKT', 'INLX', 'INM', 'INN', 'INO', 'INSG', 'INTEU', 'INTZ', 'INUV', 'INVZ', 'IOMT', 'IP', 'IPA', 'IPDN', 'IPGP', 'IPI', 'IPM', 'IPWR', 'IRD', 'IRON', 'IRS', 'IRS-WS', 'IRWD', 'ISPC', 'ISPO', 'ISPOW', 'ITP', 'ITRG', 'ITRM', 'ITUB', 'IVCAU', 'IVCB', 'IVDA', 'IVP', 'IVR', 'IVT', 'IWFL', 'IWML', 'IX', 'IXAQF', 'IZEA', 'IZM', 'J', 'JAGX', 'JAKK', 'JBK', 'JCSE', 'JDZG', 'JEF', 'JETD', 'JETU', 'JFBR', 'JFU', 'JG', 'JILL', 'JMIA', 'JMSB', 'JNVR', 'JSPR', 'JWEL', 'JXG', 'JYD', 'JZ', 'JZXN', 'K', 'KALA', 'KAR', 'KAVL', 'KEN', 'KGEI', 'KIND', 'KIRK', 'KITT', 'KLG', 'KLXE', 'KMPB', 'KNF', 'KNSL', 'KNTK', 'KNW', 'KOD', 'KOF', 'KORE', 'KPLT', 'KPRX', 'KPTI', 'KRKR', 'KRMD', 'KROS', 'KSCP', 'KTB', 'KTTA', 'KULR', 'KVYO', 'KWE', 'KXIN', 'KYTX', 'KZIA', 'KZR', 'LAC', 'LAES', 'LANDM', 'LARK', 'LAUR', 'LAW', 'LAZR', 'LBTYA', 'LBTYB', 'LBTYK', 'LC', 'LCFY', 'LCTX', 'LDER', 'LDRH', 'LDTC', 'LDWY', 'LEE', 'LEGH', 'LEGT-U', 'LEN', 'LEN-B', 'LEXX', 'LFLY', 'LFMD', 'LFST', 'LFWD', 'LGCB', 'LGF-A', 'LGF-B', 'LGHL', 'LGL', 'LGMK', 'LGND', 'LGO', 'LGVN', 'LH', 'LIBD', 'LICN', 'LIDR', 'LIDRW', 'LILA', 'LILAK', 'LINK', 'LIPO', 'LIQT', 'LITB', 'LITM', 'LIVE', 'LIXT', 'LIXTW', 'LKCO', 'LLYVA', 'LLYVK', 'LMB', 'LMFA', 'LNKB', 'LOBO', 'LOCL', 'LODE', 'LOGC', 'LPCN', 'LPRO', 'LPSN', 'LPTX', 'LRE', 'LRFC', 'LRMR', 'LSB', 'LSEA', 'LSPD', 'LSTA', 'LTBR', 'LTM', 'LTRPB', 'LTRY', 'LU', 'LUCY', 'LUCYW', 'LUNG', 'LUXH', 'LVTX', 'LVWR', 'LXEH', 'LXU', 'LYEL', 'LYRA', 'LYT', 'MAGN', 'MAMA', 'MARA', 'MARPS', 'MASI', 'MAXN', 'MAYS', 'MBAY', 'MBI', 'MBIN', 'MBINM', 'MBIO', 'MBLY', 'MBRX', 'MCAGU', 'MCBS', 'MCRB', 'MCVT', 'MDAI', 'MDRR', 'MDU', 'MDV', 'MDWD', 'MDXH', 'ME', 'MEIP', 'MESO', 'METC', 'METCB', 'MFA', 'MFH', 'MGLD', 'MGRX', 'MGYR', 'MHLA', 'MHUA', 'MIGI', 'MIND', 'MIRA', 'MITN', 'MITT', 'MKC-V', 'MKTW', 'MLEC', 'MLGO', 'MLNK', 'MLPR', 'MLTX', 'MMM', 'MMV', 'MNPR', 'MNSB', 'MNTS', 'MOBBW', 'MODD', 'MOG-B', 'MOGO', 'MOGU', 'MOVE', 'MQ', 'MRIN', 'MRK', 'MRKR', 'MRM', 'MRT', 'MRTN', 'MSBIP', 'MSGE', 'MSGM', 'MSGS', 'MSLC', 'MSPR', 'MSS', 'MSSAU', 'MSSM', 'MTA', 'MTAL', 'MTC', 'MTEM', 'MTEN', 'MTEX', 'MTH', 'MTNB', 'MTVA', 'MULN', 'MUR', 'MURA', 'MUSE', 'MUX', 'MVRL', 'MXCT', 'MYGN', 'MYND', 'MYNZ', 'MYO', 'MYSZ', 'NAAS', 'NABL', 'NAK', 'NAMS', 'NAMSW', 'NAOV', 'NATL', 'NAUT', 'NB', 'NBBK', 'NBIS', 'NBR', 'NBSTU', 'NBY', 'NCI', 'NCLH', 'NCMI', 'NCNA', 'NCPL', 'NCRA', 'NCSM', 'NCTY', 'NDRA', 'NECB', 'NEGG', 'NEN', 'NEOV', 'NEPH', 'NERV', 'NESR', 'NEUE', 'NEWP', 'NFGC', 'NFLX', 'NGNE', 'NHPBP', 'NIC', 'NISN', 'NITO', 'NIXX', 'NKGN', 'NKTR', 'NLOP', 'NLSP', 'NLY', 'NMG', 'NMM', 'NMRA', 'NMTC', 'NN', 'NNDM', 'NNVC', 'NOG', 'NOVA', 'NRDY', 'NRGV', 'NRXP', 'NSPR', 'NSSC', 'NTRB', 'NTRBW', 'NTRP', 'NTZ', 'NUKKW', 'NURO', 'NUTX', 'NUVB', 'NUWE', 'NVAX', 'NVCR', 'NVCT', 'NVDG', 'NVFY', 'NVNO', 'NVR', 'NVRI', 'NVS', 'NVST', 'NVTS', 'NVVE', 'NWE', 'NWG', 'NWGL', 'NWN', 'NXDT', 'NXL', 'NXPL', 'NXTC', 'NXTT', 'NYAX', 'NYMT', 'NYMTI', 'NYMTM', 'O', 'OABI', 'OBDC', 'OBE', 'OBLG', 'OBT', 'OCCI', 'OCCIN', 'OCEA', 'OCFCP', 'OCFT', 'OCG', 'OCGN', 'OCS', 'OCSL', 'OCTO', 'OCX', 'ODC', 'ODFL', 'ODP', 'ODV', 'OGCP', 'OGEN', 'OGI', 'OGN', 'OIS', 'OLB', 'OLLI', 'OM', 'OMEX', 'OMH', 'ONBPO', 'ONBPP', 'ONC', 'ONCO', 'ONDS', 'ONIT', 'ONL', 'ONMD', 'ONTO', 'OP', 'OPAD', 'OPAL', 'OPCH', 'OPRT', 'OPT', 'OPTN', 'OPTT', 'OPXS', 'ORC', 'ORGO', 'ORIS', 'ORKA', 'ORKT', 'ORLA', 'ORLY', 'ORMP', 'OSPN', 'OSW', 'OTLK', 'OTLY', 'OTRK', 'OUST', 'OUT', 'OVID', 'OVV', 'OWLT', 'OXLCN', 'OXSQG', 'OZ', 'PACB', 'PACK', 'PALI', 'PARAA', 'PATK', 'PAVM', 'PAVS', 'PAX', 'PAYO', 'PCAR', 'PCRX', 'PCSA', 'PDCC', 'PDLB', 'PDS', 'PDSB', 'PDYN', 'PDYNW', 'PECO', 'PEGA', 'PENG', 'PERF', 'PESI', 'PET', 'PETZ', 'PFE', 'PFTA', 'PGY', 'PGYWW', 'PHGE', 'PHGE-U', 'PHIN', 'PHIO', 'PHUN', 'PIII', 'PIK', 'PITA', 'PKBK', 'PKST', 'PLAG', 'PLBY', 'PLRZ', 'PLTD', 'PLTK', 'PLUR', 'PLX', 'PMCB', 'PMEC', 'PMN', 'PMTS', 'PNRG', 'PNST', 'POAI', 'POCI', 'PODC', 'POET', 'POLA', 'POST', 'POWW', 'PPBT', 'PRAX', 'PRE', 'PRFX', 'PRG', 'PRH', 'PRLD', 'PROC', 'PROF', 'PROK', 'PRPH', 'PRPO', 'PRQR', 'PRSO', 'PRTG', 'PSFE', 'PSHG', 'PSIX', 'PSNY', 'PSTV', 'PT', 'PTEN', 'PTGX', 'PTIX', 'PTMN', 'PTN', 'PTPI', 'PUK', 'PULM', 'PVBC', 'PWM', 'PWOD', 'PWP', 'PXLW', 'PXS', 'PYPD', 'PYT', 'PYXS', 'QBIG', 'QBTS', 'QDEL', 'QGEN', 'QH', 'QIPT', 'QLGN', 'QMCO', 'QNRX', 'QNTM', 'QOMOU', 'QQLV', 'QTTB', 'QUBT', 'QUIK', 'QULL', 'QXO', 'RAND', 'RAPT', 'RBOT', 'RCAT', 'RCEL', 'RCON', 'RDHL', 'RDIB', 'RDW', 'REAX', 'REBN', 'REE', 'REGCO', 'REGCP', 'RELI', 'RENB', 'RENEU', 'RENT', 'REPL', 'REPX', 'REVB', 'RFL', 'RGC', 'RGLS', 'RIGL', 'RILY', 'RILYL', 'RIME', 'RJF', 'RKDA', 'RKLB', 'RLI', 'RLMD', 'RMBL', 'RMTI', 'RNAC', 'RNAZ', 'ROG', 'ROIV', 'ROL', 'ROOT', 'RPAY', 'RPTX', 'RR', 'RSI', 'RSLS', 'RSSS', 'RSVRW', 'RTO', 'RTX', 'RUSHA', 'RUSHB', 'RVPH', 'RVSN', 'RYAAY', 'RZC', 'RZLT', 'SABS', 'SAFE', 'SAGE', 'SAIH', 'SANG', 'SANW', 'SATS', 'SATX', 'SAVA', 'SBEV', 'SBFG', 'SBFM', 'SBIG', 'SCCO', 'SCDL', 'SCLX', 'SCNI', 'SCNX', 'SCOR', 'SCWO', 'SCY', 'SCYX', 'SDHC', 'SDOT', 'SEAT', 'SEATW', 'SEM', 'SER', 'SERV', 'SEVN', 'SEZL', 'SF', 'SFWL', 'SGBX', 'SGD', 'SGHC', 'SGHT', 'SGLY', 'SGMA', 'SGML', 'SGMT', 'SGN', 'SHEL', 'SHEN', 'SHFS', 'SHIP', 'SHLS', 'SHLT', 'SHPH', 'SIDU', 'SIFY', 'SIGIP', 'SII', 'SILC', 'SIM', 'SINT', 'SIRI', 'SISI', 'SITC', 'SJ', 'SKE', 'SKIL', 'SKLZ', 'SKM', 'SKYE', 'SKYT', 'SKYX', 'SLDB', 'SLE', 'SLG', 'SLGL', 'SLI', 'SLMBP', 'SLNG', 'SLNH', 'SLNHP', 'SLNO', 'SLQT', 'SLRN', 'SLRX', 'SLS', 'SMCL', 'SMID', 'SMLR', 'SMR', 'SMRT', 'SMSI', 'SMTI', 'SMX', 'SNAP', 'SNAX', 'SNCR', 'SNDA', 'SNDL', 'SNES', 'SNEX', 'SNFCA', 'SNGX', 'SNOA', 'SNPX', 'SNRE', 'SNSE', 'SNTG', 'SNTI', 'SOAR', 'SOBR', 'SOFI', 'SOGP', 'SOHOB', 'SOHON', 'SOJD', 'SOLV', 'SOND', 'SONM', 'SONN', 'SOPA', 'SOS', 'SOTK', 'SOUN', 'SPCB', 'SPCE', 'SPGI', 'SPHL', 'SPHR', 'SPIR', 'SPPL', 'SPRB', 'SPRC', 'SPRO', 'SPRU', 'SPSC', 'SPXC', 'SPYU', 'SQFT', 'SQFTP', 'SQNS', 'SRAX', 'SRFM', 'SRL', 'SRM', 'SRPT', 'SRTS', 'SRZN', 'SSB', 'SSBI', 'SSKN', 'SSRM', 'SSSS', 'SSSSL', 'SST', 'STAF', 'STEC', 'STEL', 'STEM', 'STG', 'STGW', 'STHO', 'STIM', 'STKH', 'STKL', 'STNG', 'STR', 'STRM', 'STRR', 'STRRP', 'STRT', 'STRW', 'STSS', 'STXS', 'SUN', 'SUNE', 'SUNS', 'SURG', 'SVMH', 'SVRA', 'SVRE', 'SVT', 'SWBI', 'SWKH', 'SWTX', 'SWVL', 'SXTP', 'SYBX', 'SYRA', 'SYRE', 'SYRS', 'SYT', 'SYTA', 'T', 'TANH', 'TAOP', 'TAP-A', 'TARA', 'TBMCR', 'TBN', 'TC', 'TCBC', 'TCBIO', 'TCRT', 'TDG', 'TDUP', 'TDY', 'TECX', 'TEN', 'TENX', 'TFII', 'TFINP', 'TFPM', 'TFSA', 'TFX', 'TGL', 'TGS', 'TGTX', 'TH', 'THAR', 'THCH', 'THRD', 'THRY', 'THTX', 'TIGO', 'TIL', 'TIMB', 'TIRX', 'TISI', 'TIVC', 'TKLF', 'TLPH', 'TLSA', 'TLSIW', 'TLTI', 'TMCI', 'TMCWW', 'TNFA', 'TNK', 'TNON', 'TNXP', 'TOI', 'TOMZ', 'TOON', 'TOPS', 'TOVX', 'TPET', 'TPST', 'TPTA', 'TR', 'TRAW', 'TREE', 'TRIB', 'TRINZ', 'TRML', 'TRNR', 'TRST', 'TRUG', 'TRV', 'TRVG', 'TRVN', 'TSAT', 'TSBX', 'TSIBU', 'TSLA', 'TT', 'TTNP', 'TTOO', 'TTSH', 'TURN', 'TUYA', 'TWG', 'TWO', 'TXMD', 'TY-P', 'TYGO', 'UAN', 'UAVS', 'UBX', 'UCB', 'UCL', 'UGRO', 'UIS', 'UK', 'ULY', 'UNCY', 'UOKA', 'UP', 'UPC', 'UPXI', 'UROY', 'USAU', 'USEG', 'USIO', 'UTSI', 'UTZ', 'UWMC', 'UXIN', 'UZD', 'VABK', 'VANI', 'VATE', 'VCICU', 'VCIG', 'VCNX', 'VCSA', 'VEEE', 'VEON', 'VERA', 'VERB', 'VERO', 'VERU', 'VFC', 'VFF', 'VFS', 'VFSWW', 'VHC', 'VHI', 'VINC', 'VIR', 'VIRX', 'VISL', 'VIV', 'VIVK', 'VLCN', 'VLN', 'VLN-WS', 'VMAR', 'VMD', 'VMEO', 'VNOM', 'VNT', 'VOLT', 'VOXR', 'VRAR', 'VRAX', 'VRCA', 'VRDN', 'VRME', 'VRNT', 'VRPX', 'VRT', 'VS', 'VSME', 'VSSYW', 'VSTM', 'VSTS', 'VTAK', 'VTGN', 'VTLE', 'VTOL', 'VTRS', 'VTVT', 'VTYX', 'VVOS', 'VVPR', 'VYNE', 'VYX', 'W', 'WAB', 'WAI', 'WALDW', 'WATT', 'WBD', 'WBUY', 'WBX', 'WDC', 'WDS', 'WEST', 'WF', 'WFCF', 'WFG', 'WGS', 'WHLR', 'WHLRL', 'WIMI', 'WINT', 'WINV', 'WKEY', 'WKHS', 'WKSP', 'WLDS', 'WLY', 'WLYB', 'WMPN', 'WNW', 'WOR', 'WORX', 'WPC', 'WPRT', 'WRB', 'WS', 'WSO-B', 'WT', 'WTFCP', 'WTM', 'WTMAU', 'WTO', 'WVE', 'WWR', 'WYHG', 'WYTC', 'WYY', 'XAIR', 'XBIO', 'XCH', 'XCUR', 'XELB', 'XERS', 'XFOR', 'XGN', 'XIN', 'XOS', 'XPEL', 'XPER', 'XPO', 'XPON', 'XPRO', 'XRTX', 'XTKG', 'XTLB', 'XWEL', 'XXII', 'XYF', 'XYLO', 'YCBD', 'YGLD', 'YGMZ', 'YHC', 'YHNAU', 'YJ', 'YMM', 'YOSH', 'YQ', 'YSG', 'YYAI', 'ZAPP', 'ZBH', 'ZCMD', 'ZD', 'ZEPP', 'ZH', 'ZI', 'ZIMV', 'ZIONP', 'ZJYL', 'ZOOZ', 'ZTEK', 'ZURA', 'ZVRA', 'ZVSA', 'ZWS', 'ZYXI', 'ZZZ']
#stocks_to_run_on = ['ZVRA', 'ZVSA', 'ZWS', 'ZYXI', 'ZZZ']
stocks_to_run_on =  ['AFRM']
#download snp
bad_yahoo_compared_to_trading_view =  ['ABVC','ACON']

ticker = '^GSPC'
data = yf.download('^GSPC', start=start, end=end ,auto_adjust=False )
snp_df = pd.DataFrame()
snp_df['Date'] = data[('Close', ticker)].index
for k in ['High', 'Low','Open','Close', 'Volume']:
    snp_df[k] = data[(k, ticker)].values
snp_df.to_csv(os.path.join(outdir, 'snp500_yahoo.csv'))

bad_stocks = []



for ticker in  stocks_to_run_on:
    print(ticker)
    data = yf.download(ticker, start=start, end=end,auto_adjust=False)
    if len(data) == 0:
        print(f"{ticker} not found ")
        continue

    df = pd.DataFrame()
    df['Date'] = data[('Close', ticker)].index
    for k in ['High', 'Low', 'Open', 'Close', 'Volume']:
        df[k] = data[(k, ticker)].values
    try:
        df['AdjClose'] = data[('Adj Close', ticker) ].values
    except:
        df['AdjClose'] =  df['Close']

    try:
        stock_price = pd.read_parquet(os.path.join(stockdir, ticker, 'stockPrice_corrected.parquet'))
    except:
        stock_price = pd.read_parquet(os.path.join(stockdir, ticker, 'stockPrice.parquet'))

    mindate = np.min([stock_price.Date.min(), df.Date.min()])
    maxdate = np.min([stock_price.Date.max(), df.Date.max()])
    diff = (df[(df.Date >= mindate) & (df.Date <= maxdate)].Close -   stock_price[(stock_price.Date >= mindate) & (stock_price.Date <= maxdate)].close).values
    date_diff =  df[(df.Date >= mindate) & (df.Date <= maxdate)].Date
    maxdiff = np.max(np.abs(diff))

    print(f"{ticker} max diff {maxdiff}")
    if maxdiff > 1e-1:
        bad_stocks.append(ticker)
        #Display
        plt.figure()
        plt.plot(data.index, data.Close, label='yahoo close' , alpha=0.5)
        plt.plot(data.index, df['AdjClose'], label='yahoo adj close', alpha=0.5)

        plt.plot(stock_price.Date, stock_price.close, label='legacy close ', alpha=0.5)
        plt.plot(date_diff, diff)
        plt.legend()
        plt.title(ticker)
        plt.show()

    # add snp price
    snp_close_price = np.full(len(df), np.nan)
    for i, date in  enumerate(df['Date'].values):
        try:
            snp_close_price[i] = snp_df[snp_df.Date == date].Close.values[0]
        except:
            print('bad date')
    df['snp_Close'] = snp_close_price
    df.to_csv(os.path.join(os.path.join(stockdir, ticker, 'stockPrice.csv')))


    # Display
    # plt.plot(data.index, data.Close, label='yahoo close')
    #
    # #plt.plot(data.index, data.Open, label='yahoo open')
    # # plt.plot(stock_price.Date, stock_price.close, label='legacy close ')
    # # plt.plot(stock_price.Date, stock_price['1. open'], label='legacy open ')
    # plt.plot(stock_price.Date, stock_price.close, label='legacy close ')
    # #plt.plot(stock_price.Date, stock_price['1. open'], label='legacy open ')

    # plt.legend()
    # plt.title(ticker)
    # plt.show()

    df.to_csv(os.path.join(outdir, ticker + '.csv'))

print('bad_stocks')
print(bad_stocks)
#plt.show()