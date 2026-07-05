import sys; sys.path.insert(0,'.')
import config
print("OPENAI_API_KEY loaded:", "YES ("+config.OPENAI_API_KEY[:7]+"...)" if config.OPENAI_API_KEY else "NO")
print("GENERATOR default:", config.GENERATOR)
print("OPENAI_MODEL:", config.OPENAI_MODEL)