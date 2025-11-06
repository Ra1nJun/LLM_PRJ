import ScrollStack, { ScrollStackItem } from '../components/ScrollStack'
import './AboutPage.css'
import { HiDocumentSearch } from "react-icons/hi";
import { SiJfrogpipelines, SiDatadog } from "react-icons/si";

const AboutPage = () => {
    return(
        <ScrollStack>
            <ScrollStackItem itemClassName='stack-item1'>
                <div class="flex-container">
                    <div class="text-content">
                        <h2>개요</h2>
                        <p>
                            이 웹페이지는 반려견에 관한 질문에 대해 최신 정보와 검증된 답을 제공합니다.<br />
                            RAG(검색 증강 생성) 시스템으로, 단순히 저장된 답이 아닌 실제 지식문서 기반 답변을 생성합니다.<br />
                            데이터는 AI HUB의 <a href='https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71879'>반려견 성장 및 질병관련 말뭉치 데이터</a>를 사용했습니다.<br />
                            해당 데이터는 강아지나 소형 동물에 대한 논문, 서적, 수의사의 답변으로 구성되었습니다.
                        </p>
                    </div>

                    <div class="media-content">
                        <SiDatadog className='icon'/>
                    </div>
                </div>
            </ScrollStackItem>
            <ScrollStackItem itemClassName='stack-item2'>
                <div class="flex-container">
                    <div class="text-content">
                        <h2>RAG 기술 소개</h2>
                        <p>
                            RAG(Retrieval-Augmented Generation)는 AI 언어모델에 외부 문서 검색 시스템을 결합한 최신 챗봇 기술입니다.<br />
                            사용자의 질의에 맞는 실제 문서내용을 실시간으로 검색, 인공지능 답변에 반영합니다.<br />
                            훈련이 필요하던 기존 머신러닝보다 DB/문서 관리만 함으로써 유지보수가 쉽고,<br />
                            적은 컴퓨팅 자원으로도 빠른 확장이 가능합니다.<br />
                            DB/문서 근거 기반으로 답변하기 때문에 허구 답변(Hallucination) 위험 또한 감소합니다.
                        </p>
                    </div>

                    <div class="media-content">
                        <HiDocumentSearch className='icon'/>
                    </div>
                </div>
            </ScrollStackItem>
            <ScrollStackItem itemClassName='stack-item3'>
                <div class="flex-container">
                    <div class="text-content">
                        <h2>시스템 동작 원리</h2>
                        <p>
                            1. 사용자가 질문을 입력하면 Elastic 기반으로 질문을 단어들로 나눕니다.<br />
                            2. 어떤 단어가 어떤 문서에 있는지 미리 정리된 목록을 통해 검색(역색인)합니다.<br />
                            2. Elastic으로 검색된 후보 중에 임베딩 유사도 검색을 통해 질문과 '비슷한 뜻을 가진 내용, 의미'를 찾습니다.<br />
                            3. 각각 검색될 때 받은 점수를 통합해 상위 5개의 문서를 기반으로 AI가 한 번 더 걸러내고 답변을 생성합니다.
                        </p>
                    </div>

                    <div class="media-content">
                        <SiJfrogpipelines className='icon'/>
                    </div>
                </div>
            </ScrollStackItem>
        </ScrollStack>
    );
};

export default AboutPage