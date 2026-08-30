from lxml import etree

tree=etree.parse("../data/raw/aapl-20250927.htm",etree.HTMLParser())
print("parsed sucessfully")

namespaces={
    "ix":"http://www.xbrl.org/2013/inlineXBRL",
}
all_facts=tree.xpath("//*[name()='ix:nonfraction' and @name='us-gaap:ResearchAndDevelopmentExpense'] ")
#print(f"found {len(all_facts)} ix:nonfraction tags")

contexts=tree.xpath("//*[name()='xbrli:context']")
context_lookup={}
for ctx in contexts:
    ctx_id=ctx.get("id")
    #print(ctx_id,type(ctx_id))
    start=ctx.xpath(".//*[name()='xbrli:startdate']")
    end=ctx.xpath(".//*[name()='xbrli:enddate']")
    instant = ctx.xpath(".//*[name()='xbrli:instant']")
    if start and end:
        context_lookup[ctx_id]={
            "type":"duration",
            "start":start[0].text,
            "end":end[0].text
        }
    
    elif instant:
        context_lookup[ctx_id]={
            "type":"instant",
            "date":instant[0].text
        }
    else:
        context_lookup[ctx_id]={
            "type":"unknown",
            "raw":etree.tostring(ctx,encoding="unicode")
        }

#print(len(context_lookup))
print(context_lookup["c-1"])




