import email
import pandas as pd

class Preprocess():

    #Helper function for extracting email body from raw email
    def get_text_from_email(self, msg):
        parts = []
        for part in msg.walk():
            if part.get_content_type() == 'text/plain':
                parts.append( part.get_payload() )
        return ''.join(parts)

    #Helper function for seggregating the fields present in emails
    def split_email_addresses(self, line):
        if line:
            addrs = line.split(',')
            addrs = list(frozenset(map(lambda x: x.strip(), addrs)))
        else:
            addrs = None
        return addrs

    #Preprocessing emails for information extraction
    def preprocessing_emails(self, dataframe, inp_data):
        email_df = dataframe
        try:
            messages = list(map(email.message_from_string,email_df[inp_data]))
            keys = messages[0].keys()
            for key in keys:
                email_df[key] = [doc[key] for doc in messages]
            email_df['email_body'] = list(map(self.get_text_from_email, messages))
            email_df['Sender'] = email_df['From'].map(self.split_email_addresses)
            email_df['Sender'] = email_df['Sender'].str.join(',')
            email_df['Receiver'] = email_df['To'].map(self.split_email_addresses)
            email_df['Receiver'] = email_df['Receiver'].str.join(',')
            # email_df.drop(['From','To','Message-ID','Mime-Version', 'Content-Type', 'Content-Transfer-Encoding', 'X-From', 'X-To', 'X-cc', 'X-bcc', 'X-Folder', 'X-Origin', 'X-FileName'], axis=1, inplace=True)
            return email_df
        except KeyError:
            return email_df

