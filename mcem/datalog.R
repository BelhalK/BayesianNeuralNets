load("final_cat.RData")
# save.image("final_cat2.RData")


require(ggplot2)
require(gridExtra)
require(reshape2)


write.table(theo_ref_scaledbis,"theo100.txt",sep="\t",row.names=FALSE)
write.table(theo_mix25_scaledbis,"theo25.txt",sep="\t",row.names=FALSE)
write.table(theo_mix50_scaledbis_longer,"theo50.txt",sep="\t",row.names=FALSE)
write.table(theo_mix85_scaledbis,"theo85.txt",sep="\t",row.names=FALSE)
write.table(theo_mix35_scaledbis,"theo35.txt",sep="\t",row.names=FALSE)


graphConvMC_5(theo_ref,theo_mix25bis,theo_mix,theo_mix85,theo_mix35)


comparison <- rbind(
  theo_mix25_scaledbis[10:end,c(1,7,8)],
  theo_mix50_scaledbis_longer[10:nrow(theo_mix50_scaledbis_longer),c(1,7,8)],
  theo_mix35_scaledbis[10:end,c(1,7,8)],theo_mix85_scaledbis[10:end,c(1,7,8)],theo_ref_scaledbis[10:end,c(1,7,8)])


theo100beta = theo100["beta0"][:1000]
theo85beta = theo85["beta0"][:1000]
theo50beta = theo50["beta0"][:1000]
theo35beta = theo35["beta0"][:1000]
theo25beta = theo25["beta0"][:1000]

theo100gamma = theo100["gamma0"][:1000]
theo85gamma = theo85["gamma0"][:1000]
theo50gamma = theo50["gamma0"][:1000]
theo35gamma = theo35["gamma0"][:1000]
theo25gamma = theo25["gamma0"][:1000]

theo100delta = theo100["delta0"][:1000]
theo85delta = theo85["delta0"][:1000]
theo50delta = theo50["delta0"][:1000]
theo35delta = theo35["delta0"][:1000]
theo25delta = theo25["delta0"][:1000]