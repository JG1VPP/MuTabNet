_base_ = "pubtabnet.py"


template = "<html><body><table>{}</table></body></html>"

model = dict(handler=dict(revisor=dict(template=template)))

ignore = ["b"]  # in all <td></td> elements
