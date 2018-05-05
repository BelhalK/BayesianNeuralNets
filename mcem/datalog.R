load("final_cat.RData")
# save.image("final_cat2.RData")


require(ggplot2)
require(gridExtra)
require(reshape2)


write.table(theo_ref,"theo100.txt",sep="\t",row.names=FALSE)
write.table(theo_mix25bis,"theo25.txt",sep="\t",row.names=FALSE)
write.table(theo_mix,"theo50.txt",sep="\t",row.names=FALSE)
write.table(theo_mix85,"theo85.txt",sep="\t",row.names=FALSE)
write.table(theo_mix35,"theo35.txt",sep="\t",row.names=FALSE)


graphConvMC_5(theo_ref,theo_mix25bis,theo_mix,theo_mix85,theo_mix35)
