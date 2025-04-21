class Road:
    def __init__(self,roadinfo):
        super().__init__()
        self.initroad(roadinfo)

    def initroad(self,roadinfo):
        self.id = roadinfo[0]
        self.lengthr = roadinfo[1]
        self.limitv = roadinfo[2]
        self.lanenum = roadinfo[3]
        self.originc = roadinfo[4]
        self.endc = roadinfo[5]
        self.ifbiway = roadinfo[6]

        self.mintime = int(roadinfo[1]/roadinfo[2])
        self.maxcapacity = self.lanenum*self.lengthr
        self.busyrate = [[],[]]
        self.diffcarnum = [[],[]]
        self.schedule = [[],[]]

    def enterroad(self,carid,entertime,leavetime,forbackflag):
        currenttimelen = len(self.schedule[forbackflag])
        while currenttimelen < leavetime:
            self.schedule[forbackflag].append([])
            self.busyrate[forbackflag].append(0)
            self.diffcarnum[forbackflag].append(0)
        
        for time in range(entertime,leavetime):
            self.schedule[forbackflag][time-1].append(carid)
            currentcarnum = len(self.schedule[forbackflag][time-1])
            self.busyrate[forbackflag][time-1] = currentcarnum/self.maxcapacity

        return self.busyrate

Road([23,4,4,2,4,2,1])